                    ┌── Synthetic Generator
                    │
                    ├── CSV Upload
                    │
                    ├── JSON Upload
                    │
                    ├── Public Dataset
                    │
                    ├── Database
                    │
                    └── API
                         ↓
                  DATA INGESTION LAYER
                         ↓
                  VALIDATION / CLEANING
                         ↓
                  NORMALIZATION
                         ↓
                 FEATURE ENGINEERING
                         ↓
             ┌───────────┴───────────┐
             ↓                       ↓
        TABULAR PIPELINE       SEQUENCE PIPELINE
             ↓                       ↓
       KNN/LR/DT/RF             CNN/RNN/LSTM
             ↓                       ↓
             └───────────┬───────────┘
                         ↓
                 RISK CLASSIFIER
                         ↓
                 DECISION ENGINE
                         ↓
                SIMULATED ALERT
                         ↓
                    DASHBOARD