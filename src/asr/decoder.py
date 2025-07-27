import torch
import torchaudio

from torchaudio.models.decoder import ctc_decoder, download_pretrained_files


class SpeechRecognitionEngine:
    """
    ASR engine to transcribe recorded audio.
    """
    def __init__(self, model_file, token_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_file)
        self.model.eval().to(self.device)  # Move model to device (GPU if available)

        # Load decoder files and tokens
        files = download_pretrained_files("librispeech-4-gram")

        with open(token_path, 'r') as f:
            tokens = f.read().splitlines()

        self.decoder = ctc_decoder(
            lexicon=files.lexicon,
            tokens=tokens,
            lm=files.lm,
            nbest=10,
            beam_size=100,
            beam_threshold=10,
            lm_weight=3.23,
            word_score=-0.26,
        )
        print("Loaded beam search with Ken LM")

    def transcribe(self, featurizer, filename):
        """
        Transcribe audio from a file using the ASR model.
        """

        try:
            waveform, _ = torchaudio.load(filename)
            waveform = waveform.to(self.device)
            mel = featurizer(waveform).permute(0, 2, 1).to(self.device)

            with torch.inference_mode():
                out = self.model(mel)
                results = self.decoder(out.cpu())
                return " ".join(results[0][0].words).strip()
        except Exception as e:
            raise RuntimeError(f"Error during transcription: {e}")