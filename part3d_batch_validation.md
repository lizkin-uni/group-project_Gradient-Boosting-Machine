# Part 3D Batch Case Validation (Eashan)

## Status
- Workflow `inference-orchestrator`: SUCCEEDED
- Cloud Run Job `inference-job`: completed successfully
- Output file: `gs://gradient_boosting_machine/outputs/predictions.csv`

## Batch prediction summary
- Non-churn (0): 379
- Churn (1): 371
- Predicted churn rate: 49.47%

## Notes
- Initial workflow 403 issue was fixed by IAM permission update for workflow service account.
