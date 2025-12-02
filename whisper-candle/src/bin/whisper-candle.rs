use anyhow::{Error as E, Result};
use byteorder::ByteOrder as _;
use candle_core::Tensor;
use candle_transformers::models::whisper::{self as m, Config, audio};
use candle_transformers::quantized_var_builder::VarBuilder;
use clap::Parser;
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;
use whisper_candle::{
    cli::WhichModel, decoder::Decoder, decoder_model::Model, device::device, util::token_id,
};

fn main() -> Result<()> {
    // =======================================
    // Parsing Clap Argument
    // =======================================
    let args = whisper_candle::cli::Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    // =======================================
    // Choose model revision,
    // download model and save it in ~/.cache/huggingface/
    // =======================================
    let device = device(args.cpu)?;
    let (default_model, default_revision) = if args.quantized {
        ("lmz/candle-whisper", "main")
    } else {
        args.model.model_and_revision()
    };
    let default_model = default_model.to_string();
    let default_revision = default_revision.to_string();
    let (model_id, revision) = match (args.model_id, args.revision) {
        (Some(model_id), Some(revision)) => (model_id, revision),
        (Some(model_id), None) => (model_id, "main".to_string()),
        (None, Some(revision)) => (default_model, revision),
        (None, None) => (default_model, default_revision),
    };

    // =======================================
    // if user not provide wav to transcribe
    // it will download from "Narsil"
    // =======================================
    let (config_filename, tokenizer_filename, weights_filename, input) = {
        let api = Api::new()?;
        let dataset = api.dataset("Narsil/candle-examples".to_string());
        let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
        let sample = if let Some(input) = args.input {
            if let Some(sample) = input.strip_prefix("sample:") {
                dataset.get(&format!("samples_{sample}.wav"))?
            } else {
                std::path::PathBuf::from(input)
            }
        } else {
            println!(
                "No audio file submitted: Downloading https://huggingface.co/datasets/Narsil/candle_demo/blob/main/samples_jfk.wav"
            );
            dataset.get("samples_jfk.wav")?
        };
        // =======================================
        // this when program download main
        // component to run whisper
        // - config.json
        // - tokenizer.json
        // - model.json
        //
        // as shown in here
        //
        // ~/.cache/huggingface/hub/models--openai--whisper-tiny.en/snapshots/0016b25a40d2a467a0a1a273d594aa4e8110fa31> ls -la
        // ╭───┬───────────────────┬─────────┬──────────────────────────────────────────────────────────────────────────────┬───────┬─────╮
        // │ # │       name        │  type   │                                    target                                    │ reado │ ... │
        // │   │                   │         │                                                                              │ nly   │     │
        // ├───┼───────────────────┼─────────┼──────────────────────────────────────────────────────────────────────────────┼───────┼─────┤
        // │ 0 │ config.json       │ symlink │ ../../blobs/31c9f364d610705cd391e465d49df5f8e77fd868                         │ false │ ... │
        // │ 1 │ model.safetensors │ symlink │ ../../blobs/db59695928ded6043adaef491a53ef4e12da9611184d77c53baa691a60b958ad │ false │ ... │
        // │ 2 │ tokenizer.json    │ symlink │ ../../blobs/15d7bdf9ba25718ca2504eec6a8f02bc55af0a6a                         │ false │ ... │
        // ╰───┴───────────────────┴─────────┴──────────────────────────────────────────────────────────────────────────────┴───────┴─────╯
        // =======================================
        let (config, tokenizer, model) = if args.quantized {
            let ext = match args.model {
                WhichModel::TinyEn => "tiny-en",
                WhichModel::Tiny => "tiny",
                _ => unimplemented!("no quantized support for {:?}", args.model),
            };
            (
                repo.get(&format!("config-{ext}.json"))?,
                repo.get(&format!("tokenizer-{ext}.json"))?,
                repo.get(&format!("model-{ext}-q80.gguf"))?,
            )
        } else {
            let config = repo.get("config.json")?;
            let tokenizer = repo.get("tokenizer.json")?;
            let model = repo.get("model.safetensors")?;
            (config, tokenizer, model)
        };
        println!("config file: {}", config.display().to_string());
        (config, tokenizer, model, sample)
    };
    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    // =======================================
    let mel_bytes = match config.num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];

    byteorder::LittleEndian::read_f32_into(mel_bytes, &mut mel_filters);

    let (pcm_data, sample_rate) = whisper_candle::audio::pcm_decode(input)?;
    if sample_rate != m::SAMPLE_RATE as u32 {
        anyhow::bail!("input file must have a {} sampling rate", m::SAMPLE_RATE)
    }
    println!("pcm data loaded {}", pcm_data.len());
    let mel = audio::pcm_to_mel(&config, &pcm_data, &mel_filters);
    let mel_len = mel.len();
    let mel = Tensor::from_vec(
        mel,
        (1, config.num_mel_bins, mel_len / config.num_mel_bins),
        &device,
    )?;
    println!("loaded mel: {:?}", mel.dims());

    // ==============================================
    let mut model = if args.quantized {
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            &weights_filename,
            &device,
        )?;
        Model::Quantized(m::quantized_model::Whisper::load(&vb, config)?)
    } else {
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)?
        };
        Model::Normal(m::model::Whisper::load(&vb, config)?)
    };

    let language_token = match (args.model.is_multilingual(), args.language) {
        (true, None) => Some(whisper_candle::multilang::detect_language(
            &mut model, &tokenizer, &mel,
        )?),
        (false, None) => None,
        (true, Some(language)) => match token_id(&tokenizer, &format!("<|{language}|>")) {
            Ok(token_id) => Some(token_id),
            Err(_) => anyhow::bail!("language {language} is not supported"),
        },
        (false, Some(_)) => {
            anyhow::bail!("a language cannot be set for non-multilingual models")
        }
    };

    let mut dc = Decoder::new(
        model,
        tokenizer,
        args.seed,
        &device,
        language_token,
        args.task,
        args.timestamps,
        args.max_initial_timestamp_index,
        args.verbose,
    )?;

    dc.run(&mel)?;

    Ok(())
}
