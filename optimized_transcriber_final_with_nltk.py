
import os
import sys
import tempfile
import logging
import numpy as np
import whisper
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QListWidget, QTextEdit, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QProgressBar, QCheckBox, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk

nltk.download("punkt")
nltk.download('punkt_tab')
 


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class QtSignalLogHandler(logging.Handler, QObject):
    log_signal = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)


class AudioProcessor:
    def __init__(self):
        self.model_small = whisper.load_model("small")
        self.model_tiny = whisper.load_model("tiny")

    def convert_to_wav(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".wav":
            return file_path

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_channels(1)
            audio.export(tmp.name, format="wav")
            return tmp.name

    def reduce_noise(self, data, sr):
        if len(data) < sr * 2:
            return data
        return nr.reduce_noise(y=data, sr=sr)

    def get_audio_data(self, file_path):
        data, sr = sf.read(file_path)
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        return data, sr

    def transcribe(self, file_path, apply_noise=True):
        logging.info(f"Starting transcription for: {file_path}")
        wav_path = self.convert_to_wav(file_path)
        logging.info(f"Converted to WAV: {wav_path}")
        data, sr = self.get_audio_data(wav_path)
        temp_file_created = False

        if apply_noise:
            logging.info("Applying noise reduction...")
            data = self.reduce_noise(data, sr)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                sf.write(tmp_wav.name, data, sr, format='WAV')
                wav_path = tmp_wav.name
                temp_file_created = True
                logging.info(f"Noise-reduced audio written to temp file: {wav_path}")

        try:
            model_type = "tiny" if os.path.getsize(file_path) < 1 * 1024 * 1024 else "small"
            logging.info(f"Using Whisper model: {model_type}")
            model = self.model_tiny if model_type == "tiny" else self.model_small

            transcription = model.transcribe(wav_path, language="en")
            logging.info("Transcription complete.")
            return transcription["text"]
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            raise
        finally:
            if temp_file_created and os.path.exists(wav_path):
                os.remove(wav_path)
                logging.info(f"Deleted temporary file: {wav_path}")


class TranscriptionThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, processor, file_path, apply_noise):
        super().__init__()
        self.processor = processor
        self.file_path = file_path
        self.apply_noise = apply_noise

    def run(self):
        try:
            text = self.processor.transcribe(self.file_path, self.apply_noise)
            self.finished.emit(text)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimized Transcriber & Summarizer")
        self.setGeometry(100, 100, 1200, 700)
        self.processor = AudioProcessor()

        self.file_list = QListWidget()
        self.file_list.setMaximumWidth(300)

        self.transcription_output = QTextEdit()
        self.summary_output = QTextEdit()
        self.summary_output.setReadOnly(True)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(120)

        self.progress_bar = QProgressBar()
        self.noise_checkbox = QCheckBox("Apply Noise Reduction")
        self.transcribe_btn = QPushButton("Start Transcription")
        self.transcribe_btn.clicked.connect(self.start_transcription)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        self.summarize_btn = QPushButton("Summarize Transcript")
        self.summarize_btn.clicked.connect(self.summarize_text)

        self.save_transcript_btn = QPushButton("Save Transcript")
        self.save_transcript_btn.clicked.connect(self.save_transcript)

        self.save_summary_btn = QPushButton("Save Summary")
        self.save_summary_btn.clicked.connect(self.save_summary)

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(QLabel("Transcript"))
        controls_layout.addWidget(self.transcription_output)
        controls_layout.addWidget(QLabel("Summary"))
        controls_layout.addWidget(self.summary_output)
        controls_layout.addWidget(QLabel("Log Output"))
        controls_layout.addWidget(self.log_output)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addWidget(self.noise_checkbox)
        controls_layout.addWidget(self.transcribe_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.summarize_btn)
        controls_layout.addWidget(self.save_transcript_btn)
        controls_layout.addWidget(self.save_summary_btn)

        right_panel = QWidget()
        right_panel.setLayout(controls_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.file_list)
        splitter.addWidget(right_panel)

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.addWidget(splitter)

        add_file_btn = QPushButton("Add Audio File(s)")
        add_file_btn.clicked.connect(self.add_files)
        main_layout.addWidget(add_file_btn)

        self.setCentralWidget(container)

        self.log_handler = QtSignalLogHandler()
        self.log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.log_handler.log_signal.connect(self.append_log)
        logging.getLogger().addHandler(self.log_handler)

    def append_log(self, message):
        self.log_output.append(message)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Audio Files", "", "Audio Files (*.wav *.mp3 *.flac *.m4a)")
        self.file_list.addItems(files)

    def start_transcription(self):
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        file_path = selected_items[0].text()
        self.transcribe_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.thread = TranscriptionThread(self.processor, file_path, self.noise_checkbox.isChecked())
        self.thread.finished.connect(self.display_transcription)
        self.thread.error.connect(self.display_error)
        self.thread.start()

    def display_transcription(self, text):
        self.transcription_output.setText(text)
        self.summarize_btn.setEnabled(True)
        self.transcribe_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def display_error(self, error_msg):
        self.transcription_output.setText(f"Error: {error_msg}")
        self.transcribe_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def summarize_text(self):
        raw_text = self.transcription_output.toPlainText()
        if not raw_text.strip():
            self.summary_output.setPlainText("Transcript is empty.")
            return

        try:
            parser = PlaintextParser.from_string(raw_text, Tokenizer("english"))
            summarizer = TextRankSummarizer()
            summary_sentences = summarizer(parser.document, 3)
            summary = "\n".join(str(sentence) for sentence in summary_sentences)
            self.summary_output.setPlainText(summary if summary else "No summary could be generated.")
        except Exception as e:
            self.summary_output.setPlainText(f"Summary failed: {e}")

    def save_transcript(self):
        text = self.transcription_output.toPlainText()
        self.save_text_to_file(text, "Save Transcript")

    def save_summary(self):
        text = self.summary_output.toPlainText()
        self.save_text_to_file(text, "Save Summary")

    def save_text_to_file(self, text, dialog_title):
        if not text.strip():
            return

        file_path, _ = QFileDialog.getSaveFileName(self, dialog_title, "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
