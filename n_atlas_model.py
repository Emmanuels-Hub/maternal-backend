import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class NAtlasModelHandler:
    """
    Handler for the N-ATLaS GGUF model using llama.cpp.
    
    Safety guard:
    - By default, `ENABLE_LOCAL_MODEL` environment variable is set to False ("false").
    - When `ENABLE_LOCAL_MODEL` is False or model files are missing, model downloading
      from Hugging Face is SKIPPED to prevent heavy downloads on local dev machines.
    - An intelligent, context-aware rule engine provides medical fallback responses.
    """

    def __init__(
        self,
        model_id: str = "tosinamuda/N-ATLaS-GGUF",
        model_path: Optional[str] = None,
        enabled: bool = False,
        model_revision: str = "af9195261dd163807b31318b9b9d577de5ae3b7e",
        gguf_filename: str = "N-ATLaS-GGUF-Q4_K_M.gguf",
        context_size: int = 4096,
        gpu_layers: int = 0,
    ):
        self.model_id = model_id
        self.model_path = model_path or ""
        self.model_revision = model_revision
        self.gguf_filename = gguf_filename
        self.context_size = context_size
        self.gpu_layers = gpu_layers
        self.enable_local_model = enabled

        self.model = None
        self.is_loaded = False

        if self.enable_local_model:
            self._load_gguf_model()
        else:
            logger.info(
                "[N-ATLaS] ENABLE_LOCAL_MODEL is false. "
                "Model download skipped; using heuristic fallback."
            )

    def _load_gguf_model(self):
        """
        Loads the configured GGUF file locally or from the pinned Hugging Face revision.
        """
        try:
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama

            if self.model_path:
                if not os.path.isfile(self.model_path):
                    raise FileNotFoundError(
                        f"NATLAS_MODEL_PATH is not a GGUF file: {self.model_path}"
                    )
                model_path = self.model_path
            else:
                model_path = hf_hub_download(
                    repo_id=self.model_id,
                    filename=self.gguf_filename,
                    revision=self.model_revision,
                )

            self.model = Llama(
                model_path=model_path,
                n_ctx=self.context_size,
                n_gpu_layers=self.gpu_layers,
                verbose=False,
            )
            self.is_loaded = True
            logger.info(
                "[N-ATLaS] GGUF model loaded (context=%s, gpu_layers=%s).",
                self.context_size,
                self.gpu_layers,
            )
        except Exception as e:
            logger.exception(
                "[N-ATLaS] Failed to load GGUF model; using heuristic fallback: %s",
                e,
            )
            self.is_loaded = False

    def generate_doctor_explanation(
        self,
        sensor_data: Dict[str, Any],
        risk_level: str,
        patient_info: Optional[Dict[str, Any]] = None,
        language: str = "English"
    ) -> str:
        """
        Generates a maternal health doctor's note explanation using N-Atlas model or fallback engine.
        """
        if self.is_loaded and self.model:
            return self._run_natlas_inference_explanation(sensor_data, risk_level, patient_info, language)
        
        return self._fallback_doctor_explanation(sensor_data, risk_level, patient_info, language)

    def generate_chat_response(
        self,
        messages: List[Dict[str, Any]],
        user_text: str,
        image_base64: Optional[str] = None
    ) -> str:
        """
        Generates a response for health chatbot interactions.
        """
        if self.is_loaded and self.model:
            return self._run_natlas_inference_chat(messages, user_text, image_base64)
        
        return self._fallback_chat_response(user_text, image_base64)

    def translate_text(self, text: str, target_language: str) -> str:
        """
        Translates text to target language (Yoruba, Igbo, Hausa, Pidgin, English).
        """
        if self.is_loaded and self.model:
            return self._run_natlas_translation(text, target_language)
        
        return self._fallback_translation(text, target_language)

    # ─── GGUF chat-completion inference ────────────────────────────────────

    def _generate(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> str:
        if self.model is None:
            raise RuntimeError("N-ATLaS GGUF model is not loaded")

        result = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choices = result.get("choices", [])
        if not choices:
            return ""
        content = (choices[0].get("message") or {}).get("content")
        return content.strip() if isinstance(content, str) else ""

    def _run_natlas_inference_explanation(
        self,
        sensor_data: Dict[str, Any],
        risk_level: str,
        patient_info: Optional[Dict[str, Any]],
        language: str
    ) -> str:
        return self._generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You are PatDoc, a maternal wellness assistant. Explain readings "
                        f"concisely and empathetically in {language}. Do not diagnose or "
                        "replace a clinician; advise the patient to contact a healthcare "
                        "professional for concerning symptoms or readings."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Patient profile: {patient_info}\n"
                        f"Blood pressure: {sensor_data.get('bloodPressureSystolic')}/"
                        f"{sensor_data.get('bloodPressureDiastolic')} mmHg\n"
                        f"Heart rate: {sensor_data.get('heartRate')} bpm\n"
                        f"Temperature: {sensor_data.get('bodyTemp')} C\n"
                        f"SpO2: {sensor_data.get('spo2')}%\n"
                        f"ECG: {sensor_data.get('ecgResult')}\n"
                        f"Weight: {sensor_data.get('weight')} kg\n"
                        f"Risk level from the app's model: {risk_level}\n"
                        "Explain these readings without changing the supplied risk level."
                    ),
                },
            ],
            max_tokens=250,
            temperature=0.3,
        )

    def _run_natlas_inference_chat(
        self,
        messages: List[Dict[str, Any]],
        user_text: str,
        image_base64: Optional[str]
    ) -> str:
        chat_messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are PatDoc, a maternal wellness assistant. Give clear, "
                    "empathetic general information. Do not diagnose or replace a "
                    "healthcare professional. You cannot inspect images; if an image "
                    "was attached, say so and ask the user to describe it."
                ),
            }
        ]
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                chat_messages.append({"role": role, "content": content})

        if not chat_messages or chat_messages[-1].get("content") != user_text:
            chat_messages.append({"role": "user", "content": user_text})

        if image_base64:
            chat_messages[0]["content"] += " The current request includes an image."

        return self._generate(chat_messages, max_tokens=300, temperature=0.4)

    def _run_natlas_translation(self, text: str, target_language: str) -> str:
        return self._generate(
            [
                {
                    "role": "system",
                    "content": (
                        f"Translate the user's text into {target_language}. "
                        "Return only the translation and preserve medical meaning."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=200,
            temperature=0.2,
        )

    # ─── Heuristic Fallback Response Engine ─────────────────────────────────

    def _fallback_doctor_explanation(
        self,
        data: Dict[str, Any],
        risk_level: str,
        patient_info: Optional[Dict[str, Any]],
        language: str
    ) -> str:
        name = (patient_info.get("fullName") if patient_info else None) or "there"
        sys_bp = data.get("bloodPressureSystolic", 120)
        dia_bp = data.get("bloodPressureDiastolic", 80)
        temp = data.get("bodyTemp", 36.5)
        hr = data.get("heartRate", 75)
        spo2 = data.get("spo2", 98)

        bp_status = "elevated" if (sys_bp > 140 or dia_bp > 90) else "within normal range"

        lang_lower = language.lower()

        if "yoruba" in lang_lower:
            return (
                f"Ẹ káàárọ̀ {name}! Àyẹ̀wò N-Atlas rẹ fihàn pé ìwọ̀n ẹ̀jẹ̀ rẹ jẹ́ {sys_bp}/{dia_bp} mmHg ({bp_status}), "
                f"ọkàn rẹ ń lu {hr} bpm, ìgbóná ara rẹ jẹ́ {temp:.1f}°C, àti ìwọ̀n ọ̀yẹ́ ẹ̀jẹ̀ rẹ (SpO2) jẹ́ {spo2}%. "
                f"Àyẹ̀wò rẹ fihàn pe eewu rẹ jẹ́ {risk_level}. "
                f"{'Jọ̀wọ́ kàn sí dókítà rẹ káká-káká.' if risk_level == 'High Risk' else 'Àwọn àyẹ̀wò rẹ wa ní ìbáramu pẹ̀lú ilera dáradára.'} "
                "Èyí jẹ́ àyẹ̀wò N-Atlas AI — jọ̀wọ́ sọ̀rọ̀ pẹ̀lú dókítà rẹ fún àyẹ̀wò pípé."
            )
        elif "igbo" in lang_lower:
            return (
                f"Nnọọ {name}! Nnyocha N-Atlas gị na-egosi na ọbara gị dị {sys_bp}/{dia_bp} mmHg ({bp_status}), "
                f"ọnụ ọkụ obi gị bụ {hr} bpm, okpomọkụ ahụ gị bụ {temp:.1f}°C, na SpO2 bụ {spo2}%. "
                f"Tsarin N-Atlas ekwu na ọnọdụ ize ndụ gị bụ {risk_level}. "
                f"{'Biko hụ dọkịta gị ozugbo.' if risk_level == 'High Risk' else 'Ihe niile pụtara nke ọma.'} "
                "Cheta na nke a bụ N-Atlas AI — biko gakwuru dọkịta gị maka nyocha zuru ezu."
            )
        elif "hausa" in lang_lower:
            return (
                f"Barka {name}! Sakamakon gwajin N-Atlas yana nuna jinin ku yana {sys_bp}/{dia_bp} mmHg ({bp_status}), "
                f"bugun zuciya yana {hr} bpm, zafin jiki shine {temp:.1f}°C, da SpO2 yana {spo2}%. "
                f"Matakin haɗarin ku shine {risk_level}. "
                f"{'Tuntubi likita da wuri.' if risk_level == 'High Risk' else 'Sakamakon ku yana da kyau.'} "
                "Wannan N-Atlas AI ne — da fatan za ku nemi likita don cikakken gwaji."
            )
        elif "pidgin" in lang_lower:
            return (
                f"How body {name}! Your N-Atlas scan show say your blood pressure na {sys_bp}/{dia_bp} mmHg ({bp_status}), "
                f"your heart rate na {hr} bpm, body temperature na {temp:.1f}°C, and SpO2 na {spo2}%. "
                f"N-Atlas classify your risk as {risk_level}. "
                f"{'Abeg go see your doctor sharp-sharp!' if risk_level == 'High Risk' else 'Everything look steady and calm.'} "
                "Remember say this na N-Atlas AI advice — make you clear with your real doctor for full checkup."
            )
        else:
            return (
                f"Hello {name}! Based on your N-Atlas scan, your blood pressure is {bp_status} at {sys_bp}/{dia_bp} mmHg, "
                f"heart rate is {hr} bpm, body temperature is {temp:.1f}°C, and SpO2 is {spo2}%. "
                f"Our N-Atlas AI engine classified your risk level as: {risk_level}. "
                f"{'We strongly recommend consulting your obstetrician promptly for further assessment.' if risk_level == 'High Risk' else 'Your overall vital signs appear stable for your current maternal care plan.'} "
                "Note: N-Atlas is an AI diagnostic support tool. Please consult your physician for clinical diagnosis."
            )

    def _fallback_chat_response(self, user_text: str, image_base64: Optional[str]) -> str:
        text_lower = user_text.lower()
        if "blood pressure" in text_lower or "bp" in text_lower:
            return (
                "N-Atlas AI Note: During pregnancy, standard blood pressure should ideally be below 120/80 mmHg. "
                "Readings above 140/90 mmHg require clinical review to check for preeclampsia.\n\n"
                "⚠️ Always consult your physician for personalized medical evaluation."
            )
        elif "fever" in text_lower or "temperature" in text_lower:
            return (
                "N-Atlas AI Note: Normal body temperature ranges from 36.5°C to 37.2°C. "
                "A persistent body temperature above 38.0°C during pregnancy warrants immediate attention.\n\n"
                "⚠️ Please contact your healthcare provider if you experience high fever."
            )
        elif image_base64:
            return (
                "N-Atlas AI Note: I received your health image. For complete analysis of laboratory results, "
                "ultrasound scans, or skin conditions, please consult your healthcare clinician for an official review."
            )
        else:
            return (
                "Hello! I am PatDoc AI powered by the N-Atlas maternal backend. "
                "I am here to guide you with maternal wellness tips, vitals monitoring, and scan analysis.\n\n"
                "How can I assist you with your health today?"
            )

    def _fallback_translation(self, text: str, target_language: str) -> str:
        return text
