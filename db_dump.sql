BEGIN TRANSACTION;
CREATE TABLE result (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            home TEXT NOT NULL,
            visitor TEXT NOT NULL,
            date TEXT NOT NULL,
            spread REAL,
            home_score INTEGER,
            visitor_score INTEGER,
            home_expected_qb TEXT,
            visitor_expected_qb TEXT,
            home_actual_qb TEXT,
            visitor_actual_qb TEXT,
            log_reg_win_prob REAL,
            dl_win_prob REAL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            UNIQUE(home, visitor, date)
        );
INSERT INTO "result" VALUES(1,'Washington Commanders','Dallas Cowboys','2025-12-25T13:00:00-05:00',7.5,NULL,NULL,'Marcus Mariota','Dak Prescott',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:31Z','2025-12-25T20:42:31Z');
INSERT INTO "result" VALUES(2,'Minnesota Vikings','Detroit Lions','2025-12-25T16:30:00-05:00',7.0,NULL,NULL,'J J Mccarthy','Jared Goff',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:34Z','2025-12-25T20:42:34Z');
INSERT INTO "result" VALUES(3,'Kansas City Chiefs','Denver Broncos','2025-12-25T20:15:00-05:00',13.5,NULL,NULL,'Chris Oladokun','Bo Nix',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:36Z','2025-12-25T20:42:36Z');
INSERT INTO "result" VALUES(4,'Los Angeles Chargers','Houston Texans','2025-12-27T16:30:00-05:00',-1.5,NULL,NULL,'Justin Herbert','C J Stroud',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:39Z','2025-12-25T20:42:39Z');
INSERT INTO "result" VALUES(5,'Green Bay Packers','Baltimore Ravens','2025-12-27T20:00:00-05:00',-3.5,NULL,NULL,'Jordan Love','Lamar Jackson',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:42Z','2025-12-25T20:42:42Z');
INSERT INTO "result" VALUES(6,'Cincinnati Bengals','Arizona Cardinals','2025-12-28T13:00:00-05:00',-7.5,NULL,NULL,'Joe Burrow','Jacoby Brissett',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:45Z','2025-12-25T20:42:45Z');
INSERT INTO "result" VALUES(7,'Cleveland Browns','Pittsburgh Steelers','2025-12-28T13:00:00-05:00',3.0,NULL,NULL,'Shedeur Sanders','Aaron Rodgers',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:47Z','2025-12-25T20:42:47Z');
INSERT INTO "result" VALUES(8,'Tennessee Titans','New Orleans Saints','2025-12-28T13:00:00-05:00',2.5,NULL,NULL,'Cam Ward','Tyler Shough',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:50Z','2025-12-25T20:42:50Z');
INSERT INTO "result" VALUES(9,'Indianapolis Colts','Jacksonville Jaguars','2025-12-28T13:00:00-05:00',6.5,NULL,NULL,'Philip Rivers','Trevor Lawrence',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:53Z','2025-12-25T20:42:53Z');
INSERT INTO "result" VALUES(10,'Miami Dolphins','Tampa Bay Buccaneers','2025-12-28T13:00:00-05:00',6.0,NULL,NULL,'Quinn Ewers','Baker Mayfield',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:55Z','2025-12-25T20:42:55Z');
INSERT INTO "result" VALUES(11,'New York Jets','New England Patriots','2025-12-28T13:00:00-05:00',13.5,NULL,NULL,'Brady Cook','Drake Maye',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:58Z','2025-12-25T20:42:58Z');
INSERT INTO "result" VALUES(12,'Carolina Panthers','Seattle Seahawks','2025-12-28T13:00:00-05:00',7.0,NULL,NULL,'Bryce Young','Sam Darnold',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:43:01Z','2025-12-25T20:43:01Z');
INSERT INTO "result" VALUES(13,'Las Vegas Raiders','New York Giants','2025-12-28T16:05:00-05:00',1.5,NULL,NULL,'Geno Smith','Jaxson Dart',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:43:04Z','2025-12-25T20:43:04Z');
INSERT INTO "result" VALUES(14,'Buffalo Bills','Philadelphia Eagles','2025-12-28T16:25:00-05:00',-1.5,NULL,NULL,'Josh Allen','Jalen Hurts',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:43:07Z','2025-12-25T20:43:07Z');
INSERT INTO "result" VALUES(15,'San Francisco 49ers','Chicago Bears','2025-12-28T20:20:00-05:00',-3.0,NULL,NULL,'Brock Purdy','Caleb Williams',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:43:09Z','2025-12-25T20:43:09Z');
INSERT INTO "result" VALUES(16,'Atlanta Falcons','Los Angeles Rams','2025-12-29T20:15:00-05:00',7.5,NULL,NULL,'Kirk Cousins','Matthew Stafford',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:43:12Z','2025-12-25T20:43:12Z');
CREATE INDEX idx_date ON result(date)
    ;
CREATE INDEX idx_season_week ON result(season, week)
    ;
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('result',16);
COMMIT;
