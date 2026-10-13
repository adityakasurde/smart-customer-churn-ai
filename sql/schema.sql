-- =============================================================================
-- Smart Customer Churn Prediction AI - Database Schema
-- Table: prediction_logs
-- Purpose: Persist prediction history, probabilities, risk tiers, and model version
-- Constraints: No personal identifiable information (PII) is stored.
-- =============================================================================

CREATE TABLE IF NOT EXISTS prediction_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL DEFAULT '1.0.0',
    churn_probability DECIMAL(5, 4) NOT NULL,
    predicted_churn TINYINT NOT NULL,
    risk_category ENUM('Low', 'Medium', 'High') NOT NULL,
    decision_threshold DECIMAL(4, 3) NOT NULL DEFAULT 0.500,

    -- Constraints
    CONSTRAINT chk_churn_probability CHECK (churn_probability >= 0.0000 AND churn_probability <= 1.0000),
    CONSTRAINT chk_predicted_churn CHECK (predicted_churn IN (0, 1)),
    CONSTRAINT chk_decision_threshold CHECK (decision_threshold >= 0.000 AND decision_threshold <= 1.000),

    -- Indexes for reporting and history filtering
    INDEX idx_created_at (created_at DESC),
    INDEX idx_risk_category (risk_category),
    INDEX idx_predicted_churn (predicted_churn)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
