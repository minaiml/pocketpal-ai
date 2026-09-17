import type {
  CompletionTimings,
  ReasoningIntent,
} from '../../utils/completionTypes';
import type {SamplerParam, Samplers} from '../../utils/samplerParams';
import type {ServerType} from '../../utils/serverTypes';
import type {ListDerivedCaps, RemoteModelInfo} from '../../utils/types';

/** Everything a request needs about where it goes, captured per session. */
export interface RemoteEndpoint {
  url: string;
  remoteModelId: string;
  apiKey?: string;
  timeoutMs?: number;
  serverType: ServerType;
}

/** What a dialect reads off a final chunk. */
export interface FinishRead {
  timings?: CompletionTimings;
  tokensEvaluated?: number;
  tokensPredicted?: number;
}

export interface DialectRequest {
  samplers: Samplers;
  reasoning?: ReasoningIntent;
}

/**
 * How one server type is spoken to and read from. Pure description only: a
 * dialect performs no I/O, holds no state, and reads no store.
 */
export interface ServerDialect {
  type: ServerType;
  /** Send side. Present = forwarded; the value is the wire name. */
  sendNames: Partial<Record<SamplerParam, string>>;
  /** Every body key beyond the transport's own. Pure. */
  bodyExtras(req: DialectRequest): Record<string, unknown>;
  /** Final-chunk reading. Pure; never throws. */
  readFinish(chunk: unknown): FinishRead;
  /** One `/v1/models` row → list-tier caps. Pure. */
  readModelEntry(row: RemoteModelInfo | undefined): ListDerivedCaps;
  discovery: {
    /** Reachability probe path. */
    healthPath: string;
    /** `GET /props` exists and is worth probing. */
    hasProps: boolean;
    /** The llama-server router endpoints exist. */
    hasRouter: boolean;
    /** `/v1/models` rows carry caps worth a UI slot. */
    listReportsCaps: boolean;
  };
}
