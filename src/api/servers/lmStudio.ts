import type {ReasoningIntent} from '../../utils/completionTypes';
import {BASE_SEND_NAMES, openAICompatible, sendSamplers} from './base';
import type {ServerDialect} from './dialect';

/** On/off only: the LM Studio chat API ignores `reasoning_effort`. */
function reasoningExtras(
  reasoning: ReasoningIntent | undefined,
): Record<string, unknown> {
  if (!reasoning || reasoning.enabled) {
    return {};
  }
  return {chat_template_kwargs: {enable_thinking: false}};
}

export const lmStudio = {
  ...openAICompatible,
  type: 'LM Studio',
  bodyExtras: ({samplers, reasoning}) => ({
    ...sendSamplers(BASE_SEND_NAMES, samplers),
    ...reasoningExtras(reasoning),
  }),
} satisfies ServerDialect;
