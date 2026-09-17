import {toServerType} from '../../utils/serverTypes';
import type {ServerType} from '../../utils/serverTypes';
import {openAICompatible} from './base';
import type {ServerDialect} from './dialect';
import {llamaCpp} from './llamaCpp';
import {lmStudio} from './lmStudio';
import {ollama} from './ollama';
import {openaiPlatform} from './openaiPlatform';
import {vllm} from './vllm';

/**
 * One dialect per server type. Total over `ServerType`, so a new type cannot be
 * offered in the UI before it has a dialect.
 */
export const DIALECTS: Record<ServerType, ServerDialect> = {
  'llama.cpp': llamaCpp,
  'LM Studio': lmStudio,
  Ollama: ollama,
  OpenAI: openaiPlatform,
  vLLM: vllm,
  unknown: openAICompatible,
};

/**
 * The dialect for a persisted type. Anything unrecognised — a legacy empty
 * string, a free string, a case variant — speaks the base, which is what those
 * values reach today.
 */
export function dialectFor(raw: unknown): ServerDialect {
  return DIALECTS[toServerType(raw)];
}
