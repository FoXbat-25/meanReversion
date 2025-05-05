CREATE TABLE CALENDAR (
	date date not null,
	day text,
	remarks varchar(256),
	holiday bool,
	primary key(date)
)