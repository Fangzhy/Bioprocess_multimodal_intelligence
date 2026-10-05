PRAGMA foreign_keys = ON;
PRAGMA user_version = 1;

CREATE TABLE batches (
    batch_id TEXT PRIMARY KEY,
    cell_line TEXT NOT NULL,
    media_type TEXT NOT NULL,
    media_lot TEXT NOT NULL,
    bioreactor_scale_l REAL NOT NULL CHECK (bioreactor_scale_l > 0),
    seed_density_million_ml REAL NOT NULL CHECK (seed_density_million_ml > 0),
    feed_strategy TEXT NOT NULL,
    experiment_date TEXT NOT NULL
);

CREATE TABLE sensor_data (
    measurement_id INTEGER PRIMARY KEY,
    batch_id TEXT NOT NULL,
    time_hr INTEGER NOT NULL CHECK (time_hr >= 0),
    temperature_c REAL NOT NULL,
    ph REAL NOT NULL CHECK (ph BETWEEN 0 AND 14),
    do_pct REAL NOT NULL CHECK (do_pct >= 0),
    agitation_rpm REAL NOT NULL CHECK (agitation_rpm >= 0),
    air_flow_slpm REAL NOT NULL CHECK (air_flow_slpm >= 0),
    o2_flow_slpm REAL NOT NULL CHECK (o2_flow_slpm >= 0),
    co2_flow_slpm REAL NOT NULL CHECK (co2_flow_slpm >= 0),
    feed_rate_ml_hr REAL NOT NULL CHECK (feed_rate_ml_hr >= 0),
    glucose_g_l REAL NOT NULL CHECK (glucose_g_l >= 0),
    lactate_g_l REAL NOT NULL CHECK (lactate_g_l >= 0),
    viable_cell_density_million_ml REAL NOT NULL
        CHECK (viable_cell_density_million_ml >= 0),
    viability_pct REAL NOT NULL CHECK (viability_pct BETWEEN 0 AND 100),
    titer_g_l REAL NOT NULL CHECK (titer_g_l >= 0),
    FOREIGN KEY (batch_id) REFERENCES batches (batch_id),
    UNIQUE (batch_id, time_hr)
);

CREATE TABLE outcomes (
    batch_id TEXT PRIMARY KEY,
    final_titer_g_l REAL NOT NULL CHECK (final_titer_g_l >= 0),
    final_viability_pct REAL NOT NULL CHECK (final_viability_pct BETWEEN 0 AND 100),
    max_vcd_million_ml REAL NOT NULL CHECK (max_vcd_million_ml >= 0),
    quality_metric_score REAL NOT NULL CHECK (quality_metric_score BETWEEN 0 AND 100),
    FOREIGN KEY (batch_id) REFERENCES batches (batch_id)
);

CREATE TABLE text_records (
    text_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    time_hr INTEGER NOT NULL CHECK (time_hr >= 0),
    text_type TEXT NOT NULL,
    content TEXT NOT NULL,
    FOREIGN KEY (batch_id) REFERENCES batches (batch_id)
);

CREATE TABLE images (
    image_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    time_hr INTEGER NOT NULL CHECK (time_hr >= 0),
    file_path TEXT NOT NULL UNIQUE,
    image_type TEXT NOT NULL,
    is_synthetic INTEGER NOT NULL CHECK (is_synthetic IN (0, 1)),
    FOREIGN KEY (batch_id) REFERENCES batches (batch_id)
);

CREATE INDEX idx_sensor_data_batch_time
    ON sensor_data (batch_id, time_hr);

CREATE INDEX idx_text_records_batch_time
    ON text_records (batch_id, time_hr);

CREATE INDEX idx_images_batch_time
    ON images (batch_id, time_hr);
