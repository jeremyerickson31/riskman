DROP DATABASE IF EXISTS `portfolio_risk_management`;
CREATE DATABASE `portfolio_risk_management`;

GRANT ALL PRIVILEGES ON portfolio_risk_management.* TO `riskmanuser`@`localhost` IDENTIFIED BY 'abc123';

-- TABLE IS NOT NORMALIZED. ON PURPOSE
CREATE TABLE `portfolio_risk_management`.`fixed_income_securities`
(
  `id` INT(11) NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(256) NOT NULL,
  `par` DECIMAL(5,2) NOT NULL,
  `coupon` DECIMAL(4,2) NOT NULL,
  `maturity` INT(10) NOT NULL,
  `notional` INT(20) NOT NULL,
  `rating` VARCHAR(256) NOT NULL,
  `seniority` VARCHAR(256) NOT NULL,
  `portfolio_name` VARCHAR(256) NOT NULL,
   PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;
