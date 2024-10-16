use crate::{cstr, cstr_to_string, get_default_provider};
use eyre::{bail, Result};

#[derive(Debug)]
pub struct SpokenLanguageId {
    slid: *const sherpa_rs_sys::SherpaOnnxSpokenLanguageIdentification,
}

#[derive(Debug, Default)]
pub struct SpokenLanguageIdConfig {
    pub encoder: String,
    pub decoder: String,
    pub debug: Option<bool>,
    pub provider: Option<String>,
    pub num_threads: Option<i32>,
}

impl SpokenLanguageId {
    pub fn new(config: SpokenLanguageIdConfig) -> Self {
        let debug = config.debug.unwrap_or_default().into();

        let whisper = sherpa_rs_sys::SherpaOnnxSpokenLanguageIdentificationWhisperConfig {
            decoder: cstr!(config.decoder).into_raw(),
            encoder: cstr!(config.encoder).into_raw(),
            tail_paddings: 0,
        };
        let sherpa_config = sherpa_rs_sys::SherpaOnnxSpokenLanguageIdentificationConfig {
            debug,
            num_threads: config.num_threads.unwrap_or(1),
            provider: cstr!(config.provider.unwrap_or(get_default_provider())).into_raw(),
            whisper,
        };
        let slid =
            unsafe { sherpa_rs_sys::SherpaOnnxCreateSpokenLanguageIdentification(&sherpa_config) };
        Self { slid }
    }

    pub fn compute(&mut self, samples: Vec<f32>, sample_rate: i32) -> Result<String> {
        unsafe {
            let stream =
                sherpa_rs_sys::SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream(self.slid);
            sherpa_rs_sys::SherpaOnnxAcceptWaveformOffline(
                stream,
                sample_rate,
                samples.as_ptr(),
                samples.len().try_into().unwrap(),
            );
            let language_result_ptr =
                sherpa_rs_sys::SherpaOnnxSpokenLanguageIdentificationCompute(self.slid, stream);
            if language_result_ptr.is_null() || (*language_result_ptr).lang.is_null() {
                bail!("language ptr is null")
            }
            let language_ptr = (*language_result_ptr).lang;
            let language = cstr_to_string!(language_ptr);
            // Free
            sherpa_rs_sys::SherpaOnnxDestroySpokenLanguageIdentificationResult(language_result_ptr);
            sherpa_rs_sys::SherpaOnnxDestroyOfflineStream(stream);

            Ok(language)
        }
    }
}

unsafe impl Send for SpokenLanguageId {}
unsafe impl Sync for SpokenLanguageId {}

impl Drop for SpokenLanguageId {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroySpokenLanguageIdentification(self.slid);
        }
    }
}
