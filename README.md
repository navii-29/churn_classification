# Customer Churn Prediction App

Streamlit web application for predicting customer churn probability using a trained TensorFlow ANN model. Deployed from your churn classification project with `churn.csv` dataset. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/40768647/26b71042-44cc-4088-ba65-e2e8c30561c3/Ann_classification-2.ipynb?AWSAccessKeyId=ASIA2F3EMEYEYDPXV5L6&Signature=ZnIx35ggmyai5RFbEVxoZ5OhwEM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEJP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDQsWfzaEusZ%2Fpn9Tdod7Du6BxMHmbrkmOCd2%2FzEF3KFgIhAMWmxxvaCXQwbD1TWruYXwzf86Vv1oWzZ1soUqMkkw2MKvMECFwQARoMNjk5NzUzMzA5NzA1IgzILy0PcAkwo8C9EMMq0ASB3xx00r%2FsfDhUpIBvJIXtfQggtZFXmFfIR2HzWzCUJiFK8SKV2iDwlJv2cxW7euS41P%2FKhlT8xYzzL7ivYOjVeBxRbnlj1F%2BeVO7eC1x2Ve7vA5adzH0clQt3Ap14LxBNdhqlpZQAQj0HD8g05Puys%2BhGPU%2BiQ3m%2BjJumNgYROMeP6O95JSQ%2FwF0wlqrDykuy%2BtNnM5mG8VAW68mCbpAYiO1YXUowOx9uUYgttHsC0VN%2Bw6FekqQaa%2FrHNu4LJVpTYKMa%2F5s8MZP19nl%2FQjyCIqUehepbj%2FFnQmXO3eOMuKVRxjhkQrWC%2Fkg68UEssD5Jg2pLMsQRD5%2Fvx8bDck2iVzEI1TW%2FNpzRfDcmGY2RMBHAeje3ARzTFp20Ytzkz9JJLH2tP3GGMe5TLC27WLPLYWtPZr%2F3esqRCibSefK4UGYmRquF5YXlxHbs2%2Btay6xepK63jP6ijJ7rD37QO6%2FHy7YXf3JinmxMA1o60B2v%2BmkacflG2hXKBbxDh43106drtPaQ0MTbQjVga8hqppNcVCCaEjo4rl3wFOE3wPA8gHo57rTDf8IL%2F4tfHg6z6XtVktkJUPep5QPQPRCRffxp2SitP0vnPMB0SZpMuMAwS9CMWIHbej%2B563grV%2F9LhVSQ5m3873Af%2BCD7B8BdIeZlIOKX8YSB6VJyqR01HFcNcuZmGMmNQtPXAKLur6gH0kkrI95gF2ZBZHFchv368xfMukO1FT7nlLPnIgwie2md9F01I3veBMkFEz%2BK5XopkVlcNF0NQ4FEatYAiTUWf9fUMJGJxc0GOpcBH3O3eYu5w4ZVldzVCj9Bqkp3nrYe3XmVp7j%2BWLCavk7hJFrf1rEgaYM04ULdD0fPUf7FW9H6mxMXYYP9hKjf3LwZonSyTg7k7tadUIGc754aavQWHKkp8txoQHFtKEzVPmEgPqOol2%2FN1jcVXx4Ebc8%2BhmkLyL7lGlQIHYx6cBgw%2BipZBmOG8GFTst98%2FoVEnKc5RTCOzQ%3D%3D&Expires=1773227787)

Loads pre-trained model (`model.h5`) and preprocessing artifacts for real-time predictions.

## Features

- Interactive input form for customer features
- Real-time churn probability calculation
- Binary classification (churn/no churn)
- Preprocessing pipeline: LabelEncoder (gender), OneHotEncoder (geography), StandardScaler

## Required Files

Place these in the same directory:
- `model.h5` - Trained ANN model (64→32→1 neurons, sigmoid output)
- `gender_label-2.pkl` - LabelEncoder for gender
- `ohe_geo.pkl` - OneHotEncoder for geography
- `scale.pkl` - StandardScaler

## Quick Start

### 1. Install Dependencies
```bash
pip install streamlit tensorflow scikit-learn pandas numpy pickle5
```

### 2. Run the App
```bash
streamlit run app.py
```

App opens at `http://localhost:8501`.

## Input Features

| Feature | Type | Range |
|---------|------|-------|
| Geography | Select | France, Germany, Spain |
| Gender | Select | Male, Female |
| Age | Slider | 18-92 |
| Balance | Number | Any float |
| Credit Score | Number | Typically 300-850 |
| Estimated Salary | Number | Any float |
| Tenure | Slider | 0-10 years |
| Num Products | Slider | 1-4 |
| Has Credit Card | Select | 0/1 |
| Active Member | Select | 0/1 |

## Model Architecture [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/40768647/26b71042-44cc-4088-ba65-e2e8c30561c3/Ann_classification-2.ipynb?AWSAccessKeyId=ASIA2F3EMEYEYDPXV5L6&Signature=ZnIx35ggmyai5RFbEVxoZ5OhwEM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEJP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDQsWfzaEusZ%2Fpn9Tdod7Du6BxMHmbrkmOCd2%2FzEF3KFgIhAMWmxxvaCXQwbD1TWruYXwzf86Vv1oWzZ1soUqMkkw2MKvMECFwQARoMNjk5NzUzMzA5NzA1IgzILy0PcAkwo8C9EMMq0ASB3xx00r%2FsfDhUpIBvJIXtfQggtZFXmFfIR2HzWzCUJiFK8SKV2iDwlJv2cxW7euS41P%2FKhlT8xYzzL7ivYOjVeBxRbnlj1F%2BeVO7eC1x2Ve7vA5adzH0clQt3Ap14LxBNdhqlpZQAQj0HD8g05Puys%2BhGPU%2BiQ3m%2BjJumNgYROMeP6O95JSQ%2FwF0wlqrDykuy%2BtNnM5mG8VAW68mCbpAYiO1YXUowOx9uUYgttHsC0VN%2Bw6FekqQaa%2FrHNu4LJVpTYKMa%2F5s8MZP19nl%2FQjyCIqUehepbj%2FFnQmXO3eOMuKVRxjhkQrWC%2Fkg68UEssD5Jg2pLMsQRD5%2Fvx8bDck2iVzEI1TW%2FNpzRfDcmGY2RMBHAeje3ARzTFp20Ytzkz9JJLH2tP3GGMe5TLC27WLPLYWtPZr%2F3esqRCibSefK4UGYmRquF5YXlxHbs2%2Btay6xepK63jP6ijJ7rD37QO6%2FHy7YXf3JinmxMA1o60B2v%2BmkacflG2hXKBbxDh43106drtPaQ0MTbQjVga8hqppNcVCCaEjo4rl3wFOE3wPA8gHo57rTDf8IL%2F4tfHg6z6XtVktkJUPep5QPQPRCRffxp2SitP0vnPMB0SZpMuMAwS9CMWIHbej%2B563grV%2F9LhVSQ5m3873Af%2BCD7B8BdIeZlIOKX8YSB6VJyqR01HFcNcuZmGMmNQtPXAKLur6gH0kkrI95gF2ZBZHFchv368xfMukO1FT7nlLPnIgwie2md9F01I3veBMkFEz%2BK5XopkVlcNF0NQ4FEatYAiTUWf9fUMJGJxc0GOpcBH3O3eYu5w4ZVldzVCj9Bqkp3nrYe3XmVp7j%2BWLCavk7hJFrf1rEgaYM04ULdD0fPUf7FW9H6mxMXYYP9hKjf3LwZonSyTg7k7tadUIGc754aavQWHKkp8txoQHFtKEzVPmEgPqOol2%2FN1jcVXx4Ebc8%2BhmkLyL7lGlQIHYx6cBgw%2BipZBmOG8GFTst98%2FoVEnKc5RTCOzQ%3D%3D&Expires=1773227787)

From your ANN training notebook:
```
Input (12 features) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
Total params: 2,945
Trained with binary crossentropy, Adam optimizer (lr=0.01)
```

## Usage Example

1. Select customer profile (e.g., 42yo French female, high balance)
2. App preprocesses: encodes categoricals, scales numerics
3. Predicts probability >0.5 = "likely to churn"

## Deployment

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

### requirements.txt
```
streamlit==1.28.0
tensorflow==2.13.0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
pickle5==0.0.11
```

## Model Performance [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/40768647/26b71042-44cc-4088-ba65-e2e8c30561c3/Ann_classification-2.ipynb?AWSAccessKeyId=ASIA2F3EMEYEYDPXV5L6&Signature=ZnIx35ggmyai5RFbEVxoZ5OhwEM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEJP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQDQsWfzaEusZ%2Fpn9Tdod7Du6BxMHmbrkmOCd2%2FzEF3KFgIhAMWmxxvaCXQwbD1TWruYXwzf86Vv1oWzZ1soUqMkkw2MKvMECFwQARoMNjk5NzUzMzA5NzA1IgzILy0PcAkwo8C9EMMq0ASB3xx00r%2FsfDhUpIBvJIXtfQggtZFXmFfIR2HzWzCUJiFK8SKV2iDwlJv2cxW7euS41P%2FKhlT8xYzzL7ivYOjVeBxRbnlj1F%2BeVO7eC1x2Ve7vA5adzH0clQt3Ap14LxBNdhqlpZQAQj0HD8g05Puys%2BhGPU%2BiQ3m%2BjJumNgYROMeP6O95JSQ%2FwF0wlqrDykuy%2BtNnM5mG8VAW68mCbpAYiO1YXUowOx9uUYgttHsC0VN%2Bw6FekqQaa%2FrHNu4LJVpTYKMa%2F5s8MZP19nl%2FQjyCIqUehepbj%2FFnQmXO3eOMuKVRxjhkQrWC%2Fkg68UEssD5Jg2pLMsQRD5%2Fvx8bDck2iVzEI1TW%2FNpzRfDcmGY2RMBHAeje3ARzTFp20Ytzkz9JJLH2tP3GGMe5TLC27WLPLYWtPZr%2F3esqRCibSefK4UGYmRquF5YXlxHbs2%2Btay6xepK63jP6ijJ7rD37QO6%2FHy7YXf3JinmxMA1o60B2v%2BmkacflG2hXKBbxDh43106drtPaQ0MTbQjVga8hqppNcVCCaEjo4rl3wFOE3wPA8gHo57rTDf8IL%2F4tfHg6z6XtVktkJUPep5QPQPRCRffxp2SitP0vnPMB0SZpMuMAwS9CMWIHbej%2B563grV%2F9LhVSQ5m3873Af%2BCD7B8BdIeZlIOKX8YSB6VJyqR01HFcNcuZmGMmNQtPXAKLur6gH0kkrI95gF2ZBZHFchv368xfMukO1FT7nlLPnIgwie2md9F01I3veBMkFEz%2BK5XopkVlcNF0NQ4FEatYAiTUWf9fUMJGJxc0GOpcBH3O3eYu5w4ZVldzVCj9Bqkp3nrYe3XmVp7j%2BWLCavk7hJFrf1rEgaYM04ULdD0fPUf7FW9H6mxMXYYP9hKjf3LwZonSyTg7k7tadUIGc754aavQWHKkp8txoQHFtKEzVPmEgPqOol2%2FN1jcVXx4Ebc8%2BhmkLyL7lGlQIHYx6cBgw%2BipZBmOG8GFTst98%2FoVEnKc5RTCOzQ%3D%3D&Expires=1773227787)

Trained on `churn.csv` (10k samples):
- Features: CreditScore, Age, Balance, etc. + encoded Geography/Gender
- Target: Exited (binary)
- Train/test split: 80/20
- Early stopping with validation monitoring

## Troubleshooting

- **Model not found**: Ensure `model.h5` is in directory
- **CUDA errors**: Set `os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'`
- **Pickle version**: Use `pickle5` for older pickle files
- **Port busy**: `streamlit run app.py --server.port 8502`

## Project Structure

```
├── app.py                 # Streamlit app (this file)
├── model.h5              # Trained model
├── gender_label-2.pkl    # Gender encoder
├── ohe_geo.pkl           # Geography encoder
├── scale.pkl             # Scaler
├── churn.csv             # Training data
└── Ann_classification-2.ipynb  # Training notebook [file:2]
```

Your full-stack ML portfolio now includes banking API + churn predictor!
