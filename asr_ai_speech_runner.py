#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR runner using ai_speech C++ backend.
- Accepts any audio file path; converts to 16kHz mono WAV if needed.
- Returns raw transcription text (NO regex cleaning).
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import soundfile as sf

import ai_speech
import re

_ASR_TAG_RE = re.compile(r"<\|.*?\|>")  # non-greedy, matches <|zh|> <|Speech|> <|woitn|> etc.

def strip_asr_tags(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # remove tags
    text = _ASR_TAG_RE.sub(" ", text)
    # normalize whitespace
    text = " ".join(text.split()).strip()
    return text


class AudioProcessor:
    @staticmethod
    def get_audio_info(file_path: str):
        audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)
        channels = 1 if audio_data.ndim == 1 else audio_data.shape[0]
        duration = (len(audio_data) / sample_rate) if audio_data.ndim == 1 else (audio_data.shape[1] / sample_rate)
        return {
            "sample_rate": sample_rate,
            "channels": channels,
            "duration": duration,
            "format": Path(file_path).suffix.lower(),
        }

    @staticmethod
    def convert_audio_to_wav_16k_mono(input_path: str, output_path: Optional[str] = None) -> str:
        audio_data, sample_rate = librosa.load(input_path, sr=None, mono=False)

        # to mono
        if audio_data.ndim > 1:
            audio_data = librosa.to_mono(audio_data)

        # resample to 16k
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)

        # normalize
        mx = float(np.max(np.abs(audio_data))) if audio_data.size else 0.0
        if mx > 0:
            audio_data = audio_data / mx * 0.95

        if output_path is None:
            tmp = tempfile.gettempdir()
            output_path = os.path.join(tmp, f"asr_{int(time.time()*1000)}.wav")

        sf.write(output_path, audio_data, 16000, subtype="PCM_16")
        return output_path


class AISpeechASRRunner:
    """
    Thin wrapper: model_dir + device -> ai_speech.ASRPipeline -> transcribe(audio_path) -> text
    """

    def __init__(self, model_dir: Path, device: str = "CPU", model_type: str = "sensevoice"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.model_type = model_type

        self.audio_processor = AudioProcessor()
        self.asr_pipeline = None
        self.initialized = False

    def initialize(self):
        if self.initialized:
            return
        if not self.model_dir.exists():
            raise FileNotFoundError(f"ASR model dir not found: {self.model_dir}")

        self.asr_pipeline = ai_speech.ASRPipeline(
            model_dir=str(self.model_dir),
            device=self.device,
            properties={},
            model_type=self.model_type,
        )
        self.initialized = True
        print(f"[ASR] initialized: model={self.model_dir} device={self.device} type={self.model_type}")

    def release(self):
        try:
            self.asr_pipeline = None
            self.initialized = False
        finally:
            import gc
            gc.collect()

    def __call__(self, wav_path: Path) -> str:
        """
        Return transcription text only (no regex parsing).
        """
        self.initialize()

        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"wav file not found: {wav_path}")

        converted = None
        try:
            info = self.audio_processor.get_audio_info(str(wav_path))
            needs = (
                info.get("sample_rate") != 16000
                or info.get("channels", 1) != 1
                or info.get("format", "") != ".wav"
            )

            audio_for_asr = str(wav_path)
            if needs:
                converted = self.audio_processor.convert_audio_to_wav_16k_mono(str(wav_path))
                audio_for_asr = converted

            # stream collect (optional)
            chunks = []

            def _cb(text):
                chunks.append(text)
                return True

            streamer = ai_speech.create_asr_streamer(_cb)
            config = ai_speech.create_stream_config(False)

            out = self.asr_pipeline.forward(
                audio_path=str(audio_for_asr),
                config_map=config,
                streamer=streamer,
            )

            # text = out if out else "".join(chunks)
            # if not isinstance(text, str):
            #     text = str(text)

            # # minimal normalize (NOT regex tag stripping)
            # text = " ".join(text.split()).strip()
            # return text
            text = out if out else "".join(chunks)
            text = strip_asr_tags(text)
            return text


        finally:
            if converted and os.path.exists(converted):
                try:
                    os.remove(converted)
                except Exception:
                    pass
