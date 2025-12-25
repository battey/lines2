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
INSERT INTO "result" VALUES(1,'Washington Commanders','Dallas Cowboys','2025-12-25T18:00Z',8.5,NULL,NULL,'josh johnson','dak prescott',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T14:51:01Z','2025-12-25T14:51:01Z');
INSERT INTO "result" VALUES(2,'Minnesota Vikings','Detroit Lions','2025-12-25T21:30Z',7.5,NULL,NULL,'max brosmer','jared goff',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T14:51:11Z','2025-12-25T14:51:11Z');
INSERT INTO "result" VALUES(3,'Kansas City Chiefs','Denver Broncos','2025-12-26T01:15Z',13.5,NULL,NULL,'chris oladokun','bo nix',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T14:51:40Z','2025-12-25T14:51:40Z');
INSERT INTO "result" VALUES(4,'Los Angeles Chargers','Houston Texans','2025-12-27T21:30Z',-1.5,NULL,NULL,'justin herbert','c j stroud',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T14:51:51Z','2025-12-25T14:51:51Z');
INSERT INTO "result" VALUES(5,'Green Bay Packers','Baltimore Ravens','2025-12-28T01:00Z',-3.5,NULL,NULL,'jordan love','lamar jackson',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T14:51:58Z','2025-12-25T14:51:58Z');
INSERT INTO "result" VALUES(6,'Cincinnati Bengals','Arizona Cardinals','2025-12-28T18:00Z',-7.5,NULL,NULL,'joe burrow','jacoby brissett',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T14:52:15Z','2025-12-25T14:52:15Z');
INSERT INTO "result" VALUES(7,'Cleveland Browns','Pittsburgh Steelers','2025-12-28T18:00Z',3.0,NULL,NULL,'shedeur sanders','aaron rodgers',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T14:52:31Z','2025-12-25T14:52:31Z');
INSERT INTO "result" VALUES(8,'Tennessee Titans','New Orleans Saints','2025-12-28T18:00Z',2.5,NULL,NULL,'cam ward','tyler shough',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T14:52:42Z','2025-12-25T14:52:42Z');
INSERT INTO "result" VALUES(9,'Indianapolis Colts','Jacksonville Jaguars','2025-12-28T18:00Z',6.5,NULL,NULL,'philip rivers','trevor lawrence',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T14:52:56Z','2025-12-25T14:52:56Z');
INSERT INTO "result" VALUES(10,'Miami Dolphins','Tampa Bay Buccaneers','2025-12-28T18:00Z',5.5,NULL,NULL,'quinn ewers','baker mayfield',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T14:53:09Z','2025-12-25T14:53:09Z');
INSERT INTO "result" VALUES(11,'New York Jets','New England Patriots','2025-12-28T18:00Z',13.5,NULL,NULL,'brady cook','drake maye',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T14:53:19Z','2025-12-25T14:53:19Z');
INSERT INTO "result" VALUES(12,'Carolina Panthers','Seattle Seahawks','2025-12-28T18:00Z',7.0,NULL,NULL,'bryce young','sam darnold',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T14:53:42Z','2025-12-25T14:53:42Z');
INSERT INTO "result" VALUES(13,'Las Vegas Raiders','New York Giants','2025-12-28T21:05Z',1.5,NULL,NULL,'geno smith','jaxson dart',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T14:53:51Z','2025-12-25T14:53:51Z');
INSERT INTO "result" VALUES(14,'Buffalo Bills','Philadelphia Eagles','2025-12-28T21:25Z',-1.5,NULL,NULL,'josh allen','jalen hurts',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T14:54:06Z','2025-12-25T14:54:06Z');
INSERT INTO "result" VALUES(15,'San Francisco 49ers','Chicago Bears','2025-12-29T01:20Z',-3.0,NULL,NULL,'brock purdy','caleb williams',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T14:54:17Z','2025-12-25T14:54:17Z');
INSERT INTO "result" VALUES(16,'Atlanta Falcons','Los Angeles Rams','2025-12-30T01:15Z',7.5,NULL,NULL,'kirk cousins','matthew stafford',NULL,NULL,NULL,NULL,2025,17,'2025-12-25T14:54:59Z','2025-12-25T14:54:59Z');
CREATE INDEX idx_date ON result(date)
    ;
CREATE INDEX idx_season_week ON result(season, week)
    ;
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('result',16);
COMMIT;
