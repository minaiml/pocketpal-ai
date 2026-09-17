import type {ReasoningIntent} from '../../utils/completionTypes';
import {BASE_SEND_NAMES, openAICompatible, sendMap} from './base';
import type {ServerDialect} from './dialect';

/**
 * `reasoning_effort` carries the effort the caller resolved for the model id;
 * on/off alone sends nothing, because a reasoning parameter on a model that has
 * none is a 400.
 */
function reasoningExtras(
  reasoning: ReasoningIntent | undefined,
): Record<string, unknown> {
  return reasoning?.effort ? {reasoning_effort: reasoning.effort} : {};
}

export const openaiPlatform = {
  ...openAICompatible,
  type: 'OpenAI',
  ...sendMap(BASE_SEND_NAMES, reasoningExtras),
} satisfies ServerDialect;
