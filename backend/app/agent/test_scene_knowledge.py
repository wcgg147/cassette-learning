import os

def read_source():
    path = os.path.join(os.path.dirname(__file__), 'tutor_agent.py')
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def test_cold_coffee_rule():
    src = read_source()
    assert "cold coffee" in src and "iced coffee" in src, "cold coffee correction rule not found in source"
    assert "cold brew" in src, "cold brew alternative not found in source"
    print("  -> cold coffee rule: found in Hard Rules")

def test_order_accuracy_rule():
    src = read_source()
    assert "复述用户订单" in src or "只说用户实际" in src, "order accuracy rule not found in source"
    print("  -> order accuracy rule: found in Hard Rules")

def test_coffee_expressions_rule():
    src = read_source()
    assert "a latte" in src and "a cup of latte" in src, "latte expression rule not found"
    assert "half sugar" in src and "half of sugar" in src, "sugar expression rule not found"
    print("  -> coffee expressions rule: found in Hard Rules")

def test_scene_knowledge_coffee():
    src = read_source()
    assert "[Scene Knowledge]" in src, "Scene Knowledge block not found"
    assert "cappuccino" in src and "americano" in src, "coffee types not found"
    assert "room temperature" in src, "temperature expressions not found"
    print("  -> Scene Knowledge: coffee details found")

def test_few_shot_cold_coffee():
    src = read_source()
    assert "a cold coffee, please" in src, "cold coffee Few-shot user message not found"
    assert "iced coffee" in src, "iced coffee correction not found in Few-shot"
    print("  -> Few-shot: cold coffee correction example found")

def test_full_coffee_order_flow():
    src = read_source()
    assert "点单全流程" in src or "问候" in src, "order flow not found"
    assert "QR code" in src, "payment options not found"
    print("  -> Scene Knowledge: order flow found")

def main():
    tests = [
        test_cold_coffee_rule,
        test_order_accuracy_rule,
        test_coffee_expressions_rule,
        test_scene_knowledge_coffee,
        test_few_shot_cold_coffee,
        test_full_coffee_order_flow,
    ]
    passed = 0
    failed = 0
    print("Running scene item deep knowledge tests...\n")
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed")
    if failed == 0:
        print("ALL TESTS PASSED")
    return failed == 0

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
