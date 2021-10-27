USE test_db;
CREATE TABLE IF NOT EXISTS tweets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    t_key VARCHAR(255) NOT NULL,
    text VARCHAR(255) NOT NULL,
    processed_text VARCHAR(255) NOT NULL,
    polarity DOUBLE,
    created VARCHAR(255)
)