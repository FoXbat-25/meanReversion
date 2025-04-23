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
	PRIMARY KEY (SYMBOL, entry_date)
);

