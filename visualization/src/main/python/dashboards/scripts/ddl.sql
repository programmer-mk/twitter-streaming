USE myDb;
CREATE TABLE IF NOT EXISTS tweets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    t_key VARCHAR(255) NOT NULL,
    text VARCHAR(255) NOT NULL,
    processed_text VARCHAR(255) NOT NULL,
    polarity DOUBLE,
    search_term VARCHAR(255),
    created VARCHAR(255)
)