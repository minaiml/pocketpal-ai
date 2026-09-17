import type {ReasoningIntent} from '../../utils/completionTypes';
import {BASE_SEND_NAMES, openAICompatible, sendMap} from './base';
import type {ServerDialect} from './dialect';

/**
 * Ollama's `/v1` surface takes `reasoning_effort: 'none'` as a safe no-op for
 * OFF. It never takes `think: true` and never a non-`'none'` effort: both are a
 * hard 400 on a model with no thinking support. Graded effort is deferred.
 */
function reasoningExtras(
  reasoning: ReasoningIntent | undefined,
): Record<string, unknown> {
  if (!reasoning || reasoning.enabled) {
    return {};
  }
  return {reasoning_effort: 'none'};
}

export const ollama = {
  ...openAICompatible,
  type: 'Ollama',
  ...sendMap(BASE_SEND_NAMES, reasoningExtras),
} satisfies ServerDialect;
