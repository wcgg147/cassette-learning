/** Tracks whether provider ASR belongs to PCM sent after the last client final. */
export class ProviderAsrPcmBoundary {
  constructor() {
    this.clientPcmSequence = 0;
    this.lastClientFinalPcmSequence = 0;
    this.hasClientFinal = false;
    this.awaitingProviderForEmptyClientCommit = false;
    this.emptyClientCommitPcmSequence = null;
    this.providerItemPcmSequence = null;
    this.clientFinalText = '';
  }

  noteClientPcm(byteCount) {
    const bytes = Math.max(0, Number(byteCount) || 0);
    this.clientPcmSequence += bytes;
    return this.clientPcmSequence;
  }

  noteProviderStarted() {
    this.providerItemPcmSequence = this.clientPcmSequence;
  }

  noteProviderDelta() {
    // Keep the start sequence when known. A late delta for a pre-final item
    // must not become fresh merely because newer PCM has since arrived.
    if (this.providerItemPcmSequence === null) {
      this.providerItemPcmSequence = this.clientPcmSequence;
    }
  }

  noteClientFinal(text) {
    const finalText = String(text || '').trim();
    if (finalText) {
      this.lastClientFinalPcmSequence = this.clientPcmSequence;
      this.hasClientFinal = true;
      this.clientFinalText = finalText;
      this.awaitingProviderForEmptyClientCommit = false;
      this.emptyClientCommitPcmSequence = null;
    } else {
      this.awaitingProviderForEmptyClientCommit = true;
      this.emptyClientCommitPcmSequence = this.clientPcmSequence;
    }
  }

  classifyProviderCompletion({ clientCommitRequested, segmentText }) {
    const itemHasFreshPcm = this.providerItemPcmSequence !== null
      && this.providerItemPcmSequence > this.lastClientFinalPcmSequence;
    if (this.awaitingProviderForEmptyClientCommit
      && (!this.hasClientFinal || itemHasFreshPcm)
      && (this.emptyClientCommitPcmSequence === null
        || this.providerItemPcmSequence <= this.emptyClientCommitPcmSequence)) {
      return 'complete_empty_client_commit';
    }
    if (this.hasClientFinal && !itemHasFreshPcm) {
      return String(segmentText || '').trim() === this.clientFinalText
        ? 'late_duplicate'
        : 'late_mismatch';
    }
    if (clientCommitRequested || itemHasFreshPcm || !this.hasClientFinal) return 'accept';
    return String(segmentText || '').trim() === this.clientFinalText
      ? 'late_duplicate'
      : 'late_mismatch';
  }

  noteProviderCompleted({ consumeEmptyClientCommit = false } = {}) {
    this.providerItemPcmSequence = null;
    if (consumeEmptyClientCommit) this.awaitingProviderForEmptyClientCommit = false;
  }
}
