# Cloud Run model deployment

The Cloud Run image includes the CPU inference dependencies but not model
weights. Mount a Cloud Storage bucket read-only at `/models`; the application
loads model files directly from that mount and does not download them from
Hugging Face.

## Bucket layout

Upload the GGUF file and complete Hugging Face snapshot directories to the
bucket root using this layout. The leading `bucket-root/` label is explanatory;
do not create an extra `models/` directory inside the bucket:

```text
bucket-root/
  llm/N-ATLaS-GGUF-Q4_K_M.gguf
  asr/hausa/...
  asr/igbo/...
  asr/yoruba/...
```

Each ASR directory must contain the full model snapshot, including its model
configuration, tokenizer/feature-extractor files, and weights. Keep model
versions in immutable object paths and update the service configuration when
you promote a new version.

## Cloud Run configuration

Mount the bucket as a read-only Cloud Storage volume at `/models` using the
Cloud Run Volumes settings or `gcloud run services update`:

```sh
gcloud run services update SERVICE \
  --region REGION \
  --execution-environment gen2 \
  --add-volume mount-path=/models,type=cloud-storage,bucket=BUCKET_NAME,readonly=true
```

Grant the Cloud Run service identity `roles/storage.objectViewer` on the bucket.
Configure these environment variables on the service:

```text
ENABLE_LOCAL_MODEL=true
NATLAS_MODEL_PATH=/models/llm/N-ATLaS-GGUF-Q4_K_M.gguf
ENABLE_ASR_MODEL=true
NATLAS_ASR_MODEL_ROOT=/models/asr
NATLAS_ASR_DEVICE=cpu
```

Cloud Storage FUSE reads objects through the network and may use container
memory for caching. This avoids an application-managed download and a full
temporary model copy, but each new Cloud Run instance still has to read model
data from the bucket. Verify cold-start and inference performance with the
chosen Cloud Run memory/CPU allocation. The current ASR handler keeps only one
language model loaded at a time.

For local development, leave both model flags disabled or omit
`NATLAS_ASR_MODEL_ROOT` to retain the existing Hugging Face behavior.