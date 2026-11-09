import random
import time
from typing import List, Dict
from .style_parameters import StyleParameters
from .feedback_collector import FeedbackSignal


class StyleOptimizer:
    """基于反馈的风格参数优化器（简化版：梯度上升）"""

    def __init__(self, learning_rate: float = 0.05, exploration_rate: float = 0.15):
        self.lr = learning_rate
        self.epsilon = exploration_rate

        # 哪些参数可以优化（排除元数据和 user_id）
        self.optimizable_params = [
            "formality", "warmth", "humor", "directness",
            "error_tolerance", "challenge_level",
            "preferred_length", "language_mix", "example_frequency",
        ]

    def optimize(
        self,
        current_params: StyleParameters,
        feedback_signals: List[FeedbackSignal],
    ) -> StyleParameters:
        """Optimize only from explicitly attributed parameter feedback.

        Generic session-level signals are intentionally ignored.  Each signal
        must name ``context.target_parameter`` so one outcome cannot move every
        style dimension in the same direction.
        """
        if not feedback_signals:
            return current_params

        attributed = [
            signal for signal in feedback_signals
            if signal.context.get("target_parameter") in self.optimizable_params
        ]
        if not attributed:
            return current_params

        new_params = StyleParameters(**current_params.to_dict())
        for parameter in self.optimizable_params:
            signals = [s for s in attributed if s.context.get("target_parameter") == parameter]
            if not signals:
                continue
            feedback = sum(s.signal_value for s in signals) / len(signals)
            current = getattr(new_params, parameter)
            if parameter == "preferred_length":
                setattr(new_params, parameter, current + int(round(self.lr * feedback * 10)))
            elif isinstance(current, (int, float)):
                setattr(new_params, parameter, current + self.lr * feedback)

        new_params = self._clamp(new_params)

        new_params.version = current_params.version + 1
        new_params.updated_at = time.time()
        new_params.feedback_count = current_params.feedback_count + len(attributed)

        return new_params

    def _update_params(self, params: StyleParameters, feedback: float) -> StyleParameters:
        """根据反馈更新参数"""
        new_params = StyleParameters(**params.to_dict())

        for param_name in self.optimizable_params:
            current_value = getattr(params, param_name, None)
            if current_value is None:
                continue

            if param_name == "preferred_length":
                # 整数参数
                delta = int(self.lr * feedback * 5)  # 长度调整幅度大一些
                new_value = current_value + delta
                setattr(new_params, param_name, new_value)
            elif isinstance(current_value, (int, float)):
                # 连续参数
                delta = self.lr * feedback
                new_value = current_value + delta
                setattr(new_params, param_name, new_value)

        # 根据反馈调整 correction_style（离散参数）
        if feedback > 0.5:
            # 正面反馈：如果当前是 recast，保持；如果是 explicit，可以尝试 elicitation
            if params.correction_style == "explicit":
                new_params.correction_style = "elicitation"
        elif feedback < -0.3:
            # 负面反馈：切换到更直接的纠错方式
            if params.correction_style == "recast":
                new_params.correction_style = "explicit"

        return new_params

    def _explore(self, params: StyleParameters, feedback: float) -> StyleParameters:
        """小幅随机扰动"""
        param_name = random.choice(self.optimizable_params)
        current_value = getattr(params, param_name, None)

        if current_value is None:
            return params

        if param_name == "preferred_length":
            delta = random.randint(-5, 5)
            new_value = max(10, min(80, current_value + delta))
            setattr(params, param_name, new_value)
        elif isinstance(current_value, (int, float)):
            delta = random.uniform(-0.1, 0.1)
            new_value = max(0.0, min(1.0, current_value + delta))
            setattr(params, param_name, new_value)

        return params

    def _clamp(self, params: StyleParameters) -> StyleParameters:
        """边界约束"""
        params.formality = max(0.0, min(1.0, params.formality))
        params.warmth = max(0.0, min(1.0, params.warmth))
        params.humor = max(0.0, min(1.0, params.humor))
        params.directness = max(0.0, min(1.0, params.directness))
        params.error_tolerance = max(0.0, min(1.0, params.error_tolerance))
        params.challenge_level = max(0.0, min(1.0, params.challenge_level))
        params.language_mix = max(0.0, min(1.0, params.language_mix))
        params.example_frequency = max(0.0, min(1.0, params.example_frequency))
        params.preferred_length = max(10, min(80, params.preferred_length))

        # 离散参数约束
        if params.correction_style not in ("recast", "explicit", "elicitation"):
            params.correction_style = "recast"
        if params.opening_style not in ("suggest", "ask", "state"):
            params.opening_style = "suggest"
        if params.followup_style not in ("expand", "challenge", "support"):
            params.followup_style = "expand"

        return params
