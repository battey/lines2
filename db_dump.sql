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
INSERT INTO "result" VALUES(1,'Washington Commanders','Dallas Cowboys','2025-12-25T13:00:00-05:00',7.5,23,30,'Marcus Mariota','Dak Prescott',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:31Z','2025-12-31T00:04:21Z');
INSERT INTO "result" VALUES(2,'Minnesota Vikings','Detroit Lions','2025-12-25T16:30:00-05:00',7.0,23,10,'J J Mccarthy','Jared Goff',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:34Z','2025-12-31T00:04:21Z');
INSERT INTO "result" VALUES(3,'Kansas City Chiefs','Denver Broncos','2025-12-25T20:15:00-05:00',13.5,13,20,'Chris Oladokun','Bo Nix',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:36Z','2025-12-31T00:04:21Z');
INSERT INTO "result" VALUES(4,'Los Angeles Chargers','Houston Texans','2025-12-27T16:30:00-05:00',-1.5,16,20,'Justin Herbert','C J Stroud',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:39Z','2025-12-31T00:04:21Z');
INSERT INTO "result" VALUES(5,'Green Bay Packers','Baltimore Ravens','2025-12-27T20:00:00-05:00',-3.5,24,41,'Jordan Love','Lamar Jackson',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:42Z','2025-12-31T00:04:21Z');
INSERT INTO "result" VALUES(6,'Cincinnati Bengals','Arizona Cardinals','2025-12-28T13:00:00-05:00',-7.5,37,14,'Joe Burrow','Jacoby Brissett',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:45Z','2025-12-31T00:04:21Z');
INSERT INTO "result" VALUES(7,'Cleveland Browns','Pittsburgh Steelers','2025-12-28T13:00:00-05:00',3.0,13,6,'Shedeur Sanders','Aaron Rodgers',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:47Z','2025-12-31T00:04:21Z');
INSERT INTO "result" VALUES(8,'Tennessee Titans','New Orleans Saints','2025-12-28T13:00:00-05:00',2.5,26,34,'Cam Ward','Tyler Shough',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:50Z','2025-12-31T00:04:21Z');
INSERT INTO "result" VALUES(9,'Indianapolis Colts','Jacksonville Jaguars','2025-12-28T13:00:00-05:00',6.5,17,23,'Philip Rivers','Trevor Lawrence',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:53Z','2025-12-31T00:04:21Z');
INSERT INTO "result" VALUES(10,'Miami Dolphins','Tampa Bay Buccaneers','2025-12-28T13:00:00-05:00',6.0,20,17,'Quinn Ewers','Baker Mayfield',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:55Z','2025-12-31T00:04:21Z');
INSERT INTO "result" VALUES(11,'New York Jets','New England Patriots','2025-12-28T13:00:00-05:00',13.5,10,42,'Brady Cook','Drake Maye',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:42:58Z','2025-12-31T00:04:21Z');
INSERT INTO "result" VALUES(12,'Carolina Panthers','Seattle Seahawks','2025-12-28T13:00:00-05:00',7.0,10,27,'Bryce Young','Sam Darnold',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:43:01Z','2025-12-31T00:04:21Z');
INSERT INTO "result" VALUES(13,'Las Vegas Raiders','New York Giants','2025-12-28T16:05:00-05:00',1.5,10,34,'Geno Smith','Jaxson Dart',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:43:04Z','2025-12-31T00:04:21Z');
INSERT INTO "result" VALUES(14,'Buffalo Bills','Philadelphia Eagles','2025-12-28T16:25:00-05:00',-1.5,12,13,'Josh Allen','Jalen Hurts',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:43:07Z','2025-12-31T00:04:21Z');
INSERT INTO "result" VALUES(15,'San Francisco 49ers','Chicago Bears','2025-12-28T20:20:00-05:00',-3.0,42,38,'Brock Purdy','Caleb Williams',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:43:09Z','2025-12-31T00:04:21Z');
INSERT INTO "result" VALUES(16,'Atlanta Falcons','Los Angeles Rams','2025-12-29T20:15:00-05:00',7.5,27,24,'Kirk Cousins','Matthew Stafford',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T20:43:12Z','2025-12-31T00:04:21Z');
INSERT INTO "result" VALUES(17,'Tampa Bay Buccaneers','Carolina Panthers','2026-01-03T16:30:00-05:00',-2.5,NULL,NULL,'Baker Mayfield','Bryce Young',NULL,NULL,NULL,NULL,2025,18,'2025-12-31T00:04:22Z','2025-12-31T00:04:51Z');
INSERT INTO "result" VALUES(18,'San Francisco 49ers','Seattle Seahawks','2026-01-03T20:00:00-05:00',1.5,NULL,NULL,'Brock Purdy','Sam Darnold',NULL,NULL,NULL,NULL,2025,18,'2025-12-31T00:04:22Z','2025-12-31T00:04:56Z');
INSERT INTO "result" VALUES(19,'Atlanta Falcons','New Orleans Saints','2026-01-04T13:00:00-05:00',-3.5,NULL,NULL,'Kirk Cousins','Tyler Shough',NULL,NULL,NULL,NULL,2025,18,'2025-12-31T00:04:22Z','2025-12-31T00:05:02Z');
INSERT INTO "result" VALUES(20,'Cincinnati Bengals','Cleveland Browns','2026-01-04T13:00:00-05:00',-7.5,NULL,NULL,'Joe Burrow','Shedeur Sanders',NULL,NULL,NULL,NULL,2025,18,'2025-12-31T00:04:22Z','2025-12-31T00:05:07Z');
INSERT INTO "result" VALUES(21,'Minnesota Vikings','Green Bay Packers','2026-01-04T13:00:00-05:00',-6.5,NULL,NULL,'Max Brosmer','Jordan Love',NULL,NULL,NULL,NULL,2025,18,'2025-12-31T00:04:22Z','2025-12-31T00:05:43Z');
INSERT INTO "result" VALUES(22,'New York Giants','Dallas Cowboys','2026-01-04T13:00:00-05:00',3.5,NULL,NULL,'Jaxson Dart','Dak Prescott',NULL,NULL,NULL,NULL,2025,18,'2025-12-31T00:04:22Z','2025-12-31T00:05:47Z');
INSERT INTO "result" VALUES(23,'Jacksonville Jaguars','Tennessee Titans','2026-01-04T13:00:00-05:00',-12.5,NULL,NULL,'Trevor Lawrence','Cam Ward',NULL,NULL,NULL,NULL,2025,18,'2025-12-31T00:04:22Z','2025-12-31T00:05:52Z');
INSERT INTO "result" VALUES(24,'Houston Texans','Indianapolis Colts','2026-01-04T13:00:00-05:00',-10.5,NULL,NULL,'C J Stroud','Riley Leonard',NULL,NULL,NULL,NULL,2025,18,'2025-12-31T00:04:22Z','2025-12-31T00:06:00Z');
INSERT INTO "result" VALUES(25,'Buffalo Bills','New York Jets','2026-01-04T16:25:00-05:00',-7.0,NULL,NULL,'Josh Allen','Brady Cook',NULL,NULL,NULL,NULL,2025,18,'2025-12-31T00:04:22Z','2025-12-31T00:06:04Z');
INSERT INTO "result" VALUES(26,'Chicago Bears','Detroit Lions','2026-01-04T16:25:00-05:00',-3.0,NULL,NULL,'Caleb Williams','Jared Goff',NULL,NULL,NULL,NULL,2025,18,'2025-12-31T00:04:22Z','2025-12-31T00:06:08Z');
INSERT INTO "result" VALUES(27,'Denver Broncos','Los Angeles Chargers','2026-01-04T16:25:00-05:00',-12.5,NULL,NULL,'Bo Nix','Justin Herbert',NULL,NULL,NULL,NULL,2025,18,'2025-12-31T00:04:22Z','2025-12-31T00:06:12Z');
INSERT INTO "result" VALUES(28,'Las Vegas Raiders','Kansas City Chiefs','2026-01-04T16:25:00-05:00',5.5,NULL,NULL,'Geno Smith','Chris Oladokun',NULL,NULL,NULL,NULL,2025,18,'2025-12-31T00:04:22Z','2025-12-31T00:06:17Z');
INSERT INTO "result" VALUES(29,'Los Angeles Rams','Arizona Cardinals','2026-01-04T16:25:00-05:00',-7.5,NULL,NULL,'Matthew Stafford','Jacoby Brissett',NULL,NULL,NULL,NULL,2025,18,'2025-12-31T00:04:22Z','2025-12-31T00:06:22Z');
INSERT INTO "result" VALUES(30,'New England Patriots','Miami Dolphins','2026-01-04T16:25:00-05:00',-10.5,NULL,NULL,'Drake Maye','Quinn Ewers',NULL,NULL,NULL,NULL,2025,18,'2025-12-31T00:04:22Z','2025-12-31T00:06:27Z');
INSERT INTO "result" VALUES(31,'Philadelphia Eagles','Washington Commanders','2026-01-04T16:25:00-05:00',-7.0,NULL,NULL,'Jalen Hurts','Marcus Mariota',NULL,NULL,NULL,NULL,2025,18,'2025-12-31T00:04:22Z','2025-12-31T00:06:32Z');
INSERT INTO "result" VALUES(32,'Pittsburgh Steelers','Baltimore Ravens','2026-01-04T20:20:00-05:00',3.5,NULL,NULL,'Aaron Rodgers','Lamar Jackson',NULL,NULL,NULL,NULL,2025,18,'2025-12-31T00:04:22Z','2025-12-31T00:06:37Z');
CREATE INDEX idx_date ON result(date)
    ;
CREATE INDEX idx_season_week ON result(season, week)
    ;
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('result',32);
COMMIT;
