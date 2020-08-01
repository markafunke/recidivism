CREATE TABLE IF NOT EXISTS CompasTwoYear_raw (
	id int NOT NULL,
	names varchar(30) DEFAULT NULL,
	first varchar(30) DEFAULT NULL,
	last varchar(30) DEFAULT NULL,
	compas_screening_date date DEFAULT NULL,
	sex varchar(6) DEFAULT NULL,
	dob date DEFAULT NULL,
	age int DEFAULT NULL,
	age_cat varchar(20) DEFAULT NULL,
	race varchar(20) DEFAULT NULL,
	juv_fel_count int DEFAULT NULL,
	decile_score int DEFAULT NULL,
	juv_misd_count int DEFAULT NULL,
	juv_other_count int DEFAULT NULL,
	priors_count int DEFAULT NULL,
	days_b_screening_arrest int DEFAULT NULL,
	c_jail_in date DEFAULT NULL,
	c_jail_out date DEFAULT NULL,
	c_case_number varchar(20) DEFAULT NULL,
	c_offense_date date DEFAULT NULL,
	c_arrest_date date DEFAULT NULL,
	c_days_from_compas varchar(5) DEFAULT NULL,
	c_charge_degree varchar(1) DEFAULT NULL,
	c_charge_desc varchar(100) DEFAULT NULL,
	is_recid int DEFAULT NULL,
	r_case_number varchar(20) DEFAULT NULL,
	r_charge_degree varchar(5) DEFAULT NULL,
	r_days_from_arrest int DEFAULT NULL,
	r_offense_date date DEFAULT NULL,
	r_charge_desc varchar(100) DEFAULT NULL,
	r_jail_in date DEFAULT NULL,
	r_jail_out date DEFAULT NULL,
	violent_recid int DEFAULT NULL,
	is_violent_recid int DEFAULT NULL,
	vr_case_number varchar(20) DEFAULT NULL,
	vr_charge_degree varchar(5) DEFAULT NULL,
	vr_offense_date date DEFAULT NULL,
	vr_charge_desc varchar(100) DEFAULT NULL,
	type_of_assessment varchar(18) DEFAULT NULL,
	decile_score_2 int DEFAULT NULL,
	score_text varchar(6) DEFAULT NULL,
	screening_date date DEFAULT NULL,
	v_type_of_assessment varchar(18) DEFAULT NULL,
	v_decile_score int DEFAULT NULL,
	v_score_text varchar(6) DEFAULT NULL,
	v_screening_date date DEFAULT NULL,
	in_custody date DEFAULT NULL,
	out_custody date DEFAULT NULL,
	priors_count_2 int DEFAULT NULL,
	start_num int DEFAULT NULL,
	end_num int DEFAULT NULL,
	event int DEFAULT NULL,
	two_year_recid int DEFAULT NULL,
	PRIMARY KEY (id)
    );

 CREATE TABLE IF NOT EXISTS CompasTwoYearViolent_raw (  
	id int NOT NULL,
	names varchar(30) DEFAULT NULL,
	first varchar(30) DEFAULT NULL,
	last varchar(30) DEFAULT NULL,
	compas_screening_date date DEFAULT NULL,
	sex varchar(6) DEFAULT NULL,
	dob date DEFAULT NULL,
	age int DEFAULT NULL,
	age_cat varchar(20) DEFAULT NULL,
	race varchar(20) DEFAULT NULL,
	juv_fel_count int DEFAULT NULL,
	decile_score int DEFAULT NULL,
	juv_misd_count int DEFAULT NULL,
	juv_other_count int DEFAULT NULL,
	priors_count int DEFAULT NULL,
	days_b_screening_arrest int DEFAULT NULL,
	c_jail_in date DEFAULT NULL,
	c_jail_out date DEFAULT NULL,
	c_case_number varchar(20) DEFAULT NULL,
	c_offense_date date DEFAULT NULL,
	c_arrest_date date DEFAULT NULL,
	c_days_from_compas int DEFAULT NULL,
	c_charge_degree varchar(1) DEFAULT NULL,
	c_charge_desc varchar(100) DEFAULT NULL,
	is_recid int DEFAULT NULL,
	r_case_number varchar(20) DEFAULT NULL,
	r_charge_degree varchar(5) DEFAULT NULL,
	r_days_from_arrest int DEFAULT NULL,
	r_offense_date date DEFAULT NULL,
	r_charge_desc varchar(100) DEFAULT NULL,
	r_jail_in date DEFAULT NULL,
	r_jail_out date DEFAULT NULL,
	violent_recid int DEFAULT NULL,
	is_violent_recid int DEFAULT NULL,
	vr_case_number varchar(13) DEFAULT NULL,
	vr_charge_degree varchar(5) DEFAULT NULL,
	vr_offense_date date DEFAULT NULL,
	vr_charge_desc varchar(100) DEFAULT NULL,
	type_of_assessment varchar(18) DEFAULT NULL,
	decile_score_2 int DEFAULT NULL,
	score_text varchar(6) DEFAULT NULL,
	screening_date date DEFAULT NULL,
	v_type_of_assessment varchar(18) DEFAULT NULL,
	v_decile_score int DEFAULT NULL,
	v_score_text varchar(6) DEFAULT NULL,
	v_screening_date date DEFAULT NULL,
	in_custody date DEFAULT NULL,
	out_custody date DEFAULT NULL,
	priors_count_2 int DEFAULT NULL,
	start_num int DEFAULT NULL,
	end_num int DEFAULT NULL,
	event int DEFAULT NULL,
	two_year_recid int DEFAULT NULL,
	two_year_recid_2 int DEFAULT NULL,
	PRIMARY KEY (id)
    );
   

