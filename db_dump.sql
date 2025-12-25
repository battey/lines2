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
            win_prob_lr REAL,
            win_prob_dl REAL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            UNIQUE(home, visitor, date)
        );
INSERT INTO "result" VALUES(1,'Washington Commanders','Dallas Cowboys','2025-12-25T18:00Z',8.5,NULL,NULL,'Josh Johnson','Dak Prescott',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T15:25:08Z','2025-12-25T15:25:08Z');
INSERT INTO "result" VALUES(2,'Minnesota Vikings','Detroit Lions','2025-12-25T21:30Z',7.5,NULL,NULL,'Max Brosmer','Jared Goff',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T15:25:18Z','2025-12-25T15:25:18Z');
INSERT INTO "result" VALUES(3,'Kansas City Chiefs','Denver Broncos','2025-12-26T01:15Z',13.5,NULL,NULL,'Chris Oladokun','Bo Nix',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T15:25:22Z','2025-12-25T15:25:22Z');
INSERT INTO "result" VALUES(4,'Los Angeles Chargers','Houston Texans','2025-12-27T21:30Z',-1.5,NULL,NULL,'Justin Herbert','C J Stroud',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T15:25:27Z','2025-12-25T15:25:27Z');
INSERT INTO "result" VALUES(5,'Green Bay Packers','Baltimore Ravens','2025-12-28T01:00Z',-3.5,NULL,NULL,'Jordan Love','Lamar Jackson',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T15:25:33Z','2025-12-25T15:25:33Z');
INSERT INTO "result" VALUES(6,'Cincinnati Bengals','Arizona Cardinals','2025-12-28T18:00Z',-7.5,NULL,NULL,'Joe Burrow','Jacoby Brissett',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T15:25:38Z','2025-12-25T15:25:38Z');
INSERT INTO "result" VALUES(7,'Cleveland Browns','Pittsburgh Steelers','2025-12-28T18:00Z',3.0,NULL,NULL,'Shedeur Sanders','Aaron Rodgers',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T15:25:41Z','2025-12-25T15:25:41Z');
INSERT INTO "result" VALUES(8,'Tennessee Titans','New Orleans Saints','2025-12-28T18:00Z',2.5,NULL,NULL,'Cam Ward','Tyler Shough',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T15:25:46Z','2025-12-25T15:25:46Z');
INSERT INTO "result" VALUES(9,'Indianapolis Colts','Jacksonville Jaguars','2025-12-28T18:00Z',6.5,NULL,NULL,'Philip Rivers','Trevor Lawrence',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T15:25:50Z','2025-12-25T15:25:50Z');
INSERT INTO "result" VALUES(10,'Miami Dolphins','Tampa Bay Buccaneers','2025-12-28T18:00Z',5.5,NULL,NULL,'Quinn Ewers','Baker Mayfield',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T15:25:55Z','2025-12-25T15:25:55Z');
INSERT INTO "result" VALUES(11,'New York Jets','New England Patriots','2025-12-28T18:00Z',13.5,NULL,NULL,'Brady Cook','Drake Maye',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T15:25:59Z','2025-12-25T15:25:59Z');
INSERT INTO "result" VALUES(12,'Carolina Panthers','Seattle Seahawks','2025-12-28T18:00Z',7.0,NULL,NULL,'Bryce Young','Sam Darnold',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T15:26:03Z','2025-12-25T15:26:03Z');
INSERT INTO "result" VALUES(13,'Las Vegas Raiders','New York Giants','2025-12-28T21:05Z',1.5,NULL,NULL,'Geno Smith','Jaxson Dart',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T15:26:06Z','2025-12-25T15:26:06Z');
INSERT INTO "result" VALUES(14,'Buffalo Bills','Philadelphia Eagles','2025-12-28T21:25Z',-1.5,NULL,NULL,'Josh Allen','Jalen Hurts',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T15:26:11Z','2025-12-25T15:26:11Z');
INSERT INTO "result" VALUES(15,'San Francisco 49ers','Chicago Bears','2025-12-29T01:20Z',-3.0,NULL,NULL,'Brock Purdy','Caleb Williams',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T15:26:15Z','2025-12-25T15:26:15Z');
INSERT INTO "result" VALUES(16,'Atlanta Falcons','Los Angeles Rams','2025-12-30T01:15Z',7.5,NULL,NULL,'Kirk Cousins','Matthew Stafford',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T15:26:19Z','2025-12-25T15:26:19Z');
CREATE INDEX idx_date ON result(date)
    ;
CREATE INDEX idx_season_week ON result(season, week)
    ;
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('result',16);
COMMIT;
