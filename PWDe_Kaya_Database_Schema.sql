-- ==========================================================
-- PWDe Kaya? Final Database Schema (Clean Rebuild)
-- Version: 2025
-- Angela Loma (Database Admin)
-- ==========================================================


-- ========================================================
-- 1. users
-- Registered users. Authentication via Firebase (firebase_uid).
-- username & profile_photo added per SRS + teammate suggestion.
-- ========================================================
CREATE TABLE users (
  user_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  firebase_uid VARCHAR(128) NOT NULL, -- for Firebase linkage
  username VARCHAR(100) NOT NULL UNIQUE,
  first_name VARCHAR(80),
  last_name VARCHAR(80),
  email VARCHAR(150) UNIQUE,
  phone VARCHAR(30),
  profile_photo VARCHAR(500), -- URL (Firebase storage or CDN)
  role ENUM('user','business_owner','lgu_official','moderator','admin') DEFAULT 'user',
  is_active TINYINT(1) DEFAULT 1,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (user_id),
  INDEX idx_users_firebase_uid (firebase_uid)
) ENGINE=InnoDB;

-- ========================================================
-- 2. establishments
-- Stores place info; can be verified; holds counts and aggregated fields.
-- ========================================================
CREATE TABLE establishments (
  est_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  name VARCHAR(250) NOT NULL,
  overview TEXT,
  address VARCHAR(400),
  city VARCHAR(100),
  province VARCHAR(100),
  latitude DECIMAL(10,7),
  longitude DECIMAL(10,7),
  operating_hours VARCHAR(200),
  contact_number VARCHAR(50),
  contact_email VARCHAR(150),
  website VARCHAR(250),
  google_place_id VARCHAR(200),
  is_verified TINYINT(1) DEFAULT 0,
  average_rating DECIMAL(3,2) DEFAULT NULL, -- 0.00 - 5.00
  average_sentiment ENUM('positive','neutral','negative','mixed') DEFAULT NULL,
  average_confidence DECIMAL(4,3) DEFAULT NULL, -- 0.000 - 1.000
  positive_review_count INT DEFAULT 0,
  negative_review_count INT DEFAULT 0,
  mixed_review_count INT DEFAULT 0,
  total_review_count INT DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (est_id),
  INDEX idx_est_google_place (google_place_id)
) ENGINE=InnoDB;

-- ========================================================
-- 3. establishment_social_media_links (optional but useful)
-- ========================================================
CREATE TABLE establishment_social_media_links (
  social_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  est_id INT UNSIGNED NOT NULL,
  platform VARCHAR(50), -- facebook, instagram, twitter, etc.
  url VARCHAR(500),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (social_id),
  FOREIGN KEY (est_id) REFERENCES establishments(est_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ========================================================
-- 4. google_reviews
-- Raw reviews fetched from Google (original text preserved).
-- ========================================================
CREATE TABLE user_reviews (
  google_review_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  est_id INT UNSIGNED NOT NULL,
  author_name VARCHAR(200),
  author_google_id VARCHAR(200),
  original_text TEXT,
  cleaned_text TEXT,
  rating TINYINT, -- google star (if available)
  review_date DATE,
  scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (google_review_id),
  FOREIGN KEY (est_id) REFERENCES establishments(est_id) ON DELETE CASCADE,
  INDEX idx_google_est (est_id)
) ENGINE=InnoDB;

-- ========================================================
-- 5. sentiment_results
-- Output from BERT/mBERT: label (-1/0/+1), confidence...
-- ========================================================
CREATE TABLE sentiment_results (
  sentiment_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  google_review_id INT UNSIGNED NOT NULL,
  sentiment_label TINYINT, -- -1 negative, 0 neutral, 1 positive
  confidence DECIMAL(4,3), -- 0.000 - 1.000
  processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (sentiment_id),
  FOREIGN KEY (google_review_id) REFERENCES user_reviews(google_review_id) ON DELETE CASCADE,
  INDEX idx_sent_google (google_review_id)
) ENGINE=InnoDB;

-- ========================================================
-- 6. accessibility_scores
-- Aggregated computed scores per establishment
-- ========================================================
CREATE TABLE accessibility_scores (
  score_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  est_id INT UNSIGNED NOT NULL,
  computed_score DECIMAL(3,2) CHECK (computed_score >= 0.00 AND computed_score <= 5.00), -- 0.00 - 5.00
  source ENUM('nlp','user','combined') DEFAULT 'combined',
  computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (score_id),
  FOREIGN KEY (est_id) REFERENCES establishments(est_id) ON DELETE CASCADE,
  INDEX idx_score_est (est_id)
) ENGINE=InnoDB;

-- ========================================================
-- 7. guests
-- Represents a reviewer (may or may not be a registered user)
-- If registered user, user_id links; if not, user_id IS NULL.
-- ========================================================
CREATE TABLE guests (
  guest_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  user_id INT UNSIGNED NULL, -- NULL for unregistered visitors
  contact_number VARCHAR(30),
  is_pwd TINYINT(1) DEFAULT 0,
  number_of_reviews INT DEFAULT 0,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (guest_id),
  FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL,
  INDEX idx_guest_user (user_id)
) ENGINE=InnoDB;

-- ========================================================
-- 8. guest_disabilities
-- Optional table listing disabilities for guests who are PWDs.
-- ========================================================
CREATE TABLE guest_disabilities (
  disability_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  guest_id INT UNSIGNED NOT NULL,
  disability_type VARCHAR(100),
  disability_details VARCHAR(300),
  PRIMARY KEY (disability_id),
  FOREIGN KEY (guest_id) REFERENCES guests(guest_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ========================================================
-- 9. reviews
-- In-app user reviews (user-submitted). This is the primary source for user contributions.
-- ========================================================
CREATE TABLE reviews (
  review_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  guest_id INT UNSIGNED NOT NULL,
  est_id INT UNSIGNED NOT NULL,
  denorm_est_name VARCHAR(250), -- denormalized for quick access (kept in sync by app/backend)
  review_content TEXT,
  rating DECIMAL(2,1), -- 1.0 - 5.0
  sentiment ENUM('positive','neutral','negative','mixed') DEFAULT NULL, -- optional prefilled by quick ML
  sentiment_confidence DECIMAL(4,3) DEFAULT NULL,
  status ENUM('pending','approved','rejected','flagged') DEFAULT 'pending',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (review_id),
  FOREIGN KEY (guest_id) REFERENCES guests(guest_id) ON DELETE CASCADE,
  FOREIGN KEY (est_id) REFERENCES establishments(est_id) ON DELETE CASCADE,
  INDEX idx_rev_guest (guest_id),
  INDEX idx_rev_est (est_id)
) ENGINE=InnoDB;

-- ========================================================
-- 10. review_accessibility_keywords
-- Keywords detected by ML (mobility, visual, hearing, general, etc.)
-- ========================================================
CREATE TABLE review_accessibility_keywords (
  keyword_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  review_id INT UNSIGNED NOT NULL,
  keyword VARCHAR(200),
  keyword_category VARCHAR(80),
  detection_confidence DECIMAL(4,3),
  PRIMARY KEY (keyword_id),
  FOREIGN KEY (review_id) REFERENCES reviews(review_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ========================================================
-- 11. review_assessments
-- Moderator assessment of reviews (one assessment per review).
-- ========================================================
CREATE TABLE review_assessments (
  assessment_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  review_id INT UNSIGNED NOT NULL UNIQUE, -- one assessment per review
  moderator_user_id INT UNSIGNED NULL, -- user_id of moderator; NULL if not yet assessed
  status ENUM('pending','assessed','flagged','approved_for_sentiment_analysis','approved_for_posting','rejected_for_posting','rejected') DEFAULT 'pending',
  is_flagged TINYINT(1) DEFAULT 0,
  flag_reason VARCHAR(500),
  moderator_remarks TEXT,
  assessed_at TIMESTAMP NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (assessment_id),
  FOREIGN KEY (review_id) REFERENCES reviews(review_id) ON DELETE CASCADE,
  FOREIGN KEY (moderator_user_id) REFERENCES users(user_id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- ========================================================
-- 12. request_forms
-- Base requests for reports (establishment or city)
-- ========================================================
CREATE TABLE request_forms (
  request_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  requester_user_id INT UNSIGNED NOT NULL, -- must be registered
  requester_name VARCHAR(200),
  requester_position VARCHAR(200),
  requester_email VARCHAR(200),
  requester_contact VARCHAR(50),
  report_period_from VARCHAR(20), -- e.g., "01/2025" or "MM/YYYY"
  report_period_to VARCHAR(20),
  purpose TEXT,
  remarks TEXT,
  request_type ENUM('establishment_report','city_report') DEFAULT 'establishment_report',
  status ENUM('pending','under_review','approved','rejected') DEFAULT 'pending',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (request_id),
  FOREIGN KEY (requester_user_id) REFERENCES users(user_id) ON DELETE CASCADE,
  INDEX idx_request_user (requester_user_id)
) ENGINE=InnoDB;

-- ========================================================
-- 13. establishment_report_forms (extension of request_forms)
-- ========================================================
CREATE TABLE establishment_report_forms (
  establishment_report_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  request_id INT UNSIGNED NOT NULL UNIQUE,
  est_id INT UNSIGNED NULL,
  est_name VARCHAR(300),
  business_permit_number VARCHAR(200),
  PRIMARY KEY (establishment_report_id),
  FOREIGN KEY (request_id) REFERENCES request_forms(request_id) ON DELETE CASCADE,
  FOREIGN KEY (est_id) REFERENCES establishments(est_id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- ========================================================
-- 14. city_report_forms (extension of request_forms)
-- ========================================================
CREATE TABLE city_report_forms (
  city_report_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  request_id INT UNSIGNED NOT NULL UNIQUE,
  city_name VARCHAR(200),
  lgu_office VARCHAR(300),
  PRIMARY KEY (city_report_id),
  FOREIGN KEY (request_id) REFERENCES request_forms(request_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ========================================================
-- 15. request_assessments
-- Moderator assessment of report requests (one per request)
-- ========================================================
CREATE TABLE request_assessments (
  request_assessment_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  request_id INT UNSIGNED NOT NULL UNIQUE,
  moderator_user_id INT UNSIGNED NULL,
  status ENUM('pending','approved','rejected') DEFAULT 'pending',
  moderator_remarks TEXT,
  assessed_at TIMESTAMP NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (request_assessment_id),
  FOREIGN KEY (request_id) REFERENCES request_forms(request_id) ON DELETE CASCADE,
  FOREIGN KEY (moderator_user_id) REFERENCES users(user_id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- ========================================================
-- 16. generated_reports (optional)
-- Track generated report files (pdf/excel)
-- ========================================================
CREATE TABLE generated_reports (
  report_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  request_id INT UNSIGNED NOT NULL,
  report_file_url VARCHAR(500),
  report_format ENUM('pdf','excel','csv') DEFAULT 'pdf',
  generated_by_user_id INT UNSIGNED NULL,
  generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (report_id),
  FOREIGN KEY (request_id) REFERENCES request_forms(request_id) ON DELETE CASCADE,
  FOREIGN KEY (generated_by_user_id) REFERENCES users(user_id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- ========================================================
-- 17. flagged_reviews (simpler log for flags on user-submitted reviews)
-- ========================================================
CREATE TABLE flagged_reviews (
  flag_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
  review_id INT UNSIGNED NOT NULL,
  flagged_by_user_id INT UNSIGNED NULL,
  reason VARCHAR(500),
  status ENUM('pending','resolved','dismissed') DEFAULT 'pending',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (flag_id),
  FOREIGN KEY (review_id) REFERENCES reviews(review_id) ON DELETE CASCADE,
  FOREIGN KEY (flagged_by_user_id) REFERENCES users(user_id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- ========================================================
-- Useful convenience views (optional) - Example: establishment_summary
-- This view aggregates the counts for quick retrieval.
-- ========================================================
DROP VIEW IF EXISTS establishment_summary;
CREATE VIEW establishment_summary AS
SELECT
  e.est_id,
  e.name,
  e.city,
  e.province,
  e.total_review_count,
  COALESCE(s.computed_score, NULL) AS computed_score,
  e.average_sentiment,
  e.average_confidence,
  e.positive_review_count,
  e.negative_review_count,
  e.mixed_review_count
FROM establishments e
LEFT JOIN (
  SELECT est_id, MAX(computed_score) AS computed_score
  FROM accessibility_scores
  GROUP BY est_id
) s ON s.est_id = e.est_id;