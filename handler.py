from src.rp_handler import run_whisper_job
import runpod

runpod.serverless.start({"handler": run_whisper_job})
