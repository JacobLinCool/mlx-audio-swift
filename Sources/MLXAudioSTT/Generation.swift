import MLX

public struct STTGenerateParameters: Sendable {
    public let maxTokens: Int
    public let temperature: Float
    public let topP: Float
    public let topK: Int
    public let verbose: Bool
    public let language: String?
    public let chunkDuration: Float
    public let minChunkDuration: Float
    public let repetitionPenalty: Float
    public let repetitionContextSize: Int
    /// KV-cache quantization bits; `nil` keeps model precision.
    public let kvBits: Int?
    /// Group size for KV-cache quantization.
    public let kvGroupSize: Int
    /// Cache offset that must be exceeded before the KV cache is quantized.
    public let quantizedKVStart: Int
    /// Decoding instruction/prompt. Models that support prompting (e.g. MOSS-Transcribe-Diarize,
    /// where it carries hotword hints and custom instructions) use this in place of their default
    /// prompt; models without prompt support ignore it. `nil` keeps the model default.
    public let prompt: String?
    /// When > 0, capture the top-K log-probabilities for every generated token
    /// (see `STTOutput.tokenLogprobs`). `0` disables capture and adds no overhead.
    public let logprobsTopK: Int

    public init(
        maxTokens: Int = 8192,
        temperature: Float = 0.0,
        topP: Float = 0.95,
        topK: Int = 0,
        verbose: Bool = false,
        language: String? = nil,
        chunkDuration: Float = 1200.0,
        minChunkDuration: Float = 1.0,
        repetitionPenalty: Float = 1.0,
        repetitionContextSize: Int = 32,
        kvBits: Int? = nil,
        kvGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        prompt: String? = nil,
        logprobsTopK: Int = 0
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.verbose = verbose
        self.language = language
        self.chunkDuration = chunkDuration
        self.minChunkDuration = minChunkDuration
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.prompt = prompt
        self.logprobsTopK = logprobsTopK
    }
}

public protocol STTGenerationModel: AnyObject {
    var defaultGenerationParameters: STTGenerateParameters { get }

    func generate(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> STTOutput

    func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error>
}

public extension STTGenerationModel {
    func generate(
        audio: MLXArray,
        generationParameters: STTGenerateParameters? = nil
    ) -> STTOutput {
        generate(audio: audio, generationParameters: generationParameters ?? defaultGenerationParameters)
    }

    func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters? = nil
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        generateStream(audio: audio, generationParameters: generationParameters ?? defaultGenerationParameters)
    }
}
