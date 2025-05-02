CREATE TABLE trade_book (
    symbol TEXT NOT NULL,
    entry_date DATE NOT NULL,
    entry_price FLOAT NOT NULL,
    exit_date DATE,
    exit_price FLOAT,
    status TEXT NOT NULL DEFAULT 'OPEN',
    strategy TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    pl FLOAT GENERATED ALWAYS AS (exit_price - entry_price) STORED,
    pl_percent FLOAT GENERATED ALWAYS AS ((exit_price - entry_price) / entry_price * 100) STORED,
	PRIMARY KEY (symbol, entry_date)
);

