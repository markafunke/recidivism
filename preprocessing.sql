-- limit to only columns needed in analysis
-- Limited data in same manner as propublica analysis
-- If the charge date of a defendants Compas scored crime was not within 30 days from when the person was arrested, we assume that because of data quality reasons, that we do not have the right offense.
-- We coded the recidivist flag -- is_recid -- to be -1 if we could not find a compas case at all.
-- In a similar vein, ordinary traffic offenses -- those with a c_charge_degree of 'O' -- will not result in Jail time are removed (only two of them).

create table compastwoyear as
select id, first, last, dob, sex, age, age_cat, race, juv_fel_count, decile_score, juv_misd_count, juv_other_count
, priors_count, days_b_screening_arrest, c_jail_in, c_jail_out, c_charge_degree, c_charge_desc, is_recid, r_charge_degree
, r_charge_desc, score_text, two_year_recid
from compastwoyear_raw
where (days_b_screening_arrest <= 30) and 
      (days_b_screening_arrest >= -30) and
      (is_recid != -1) and
      (c_charge_degree != 'O');

create table compastwoyear_violent as
select id, first, last, dob, sex, age, age_cat, race, juv_fel_count, decile_score, juv_misd_count, juv_other_count
, priors_count, days_b_screening_arrest, c_jail_in, c_jail_out, c_charge_degree, c_charge_desc, is_recid, vr_charge_degree
, vr_charge_desc, score_text, two_year_recid, violent_recid, is_violent_recid, v_decile_score, v_score_text
from compastwoyearviolent_raw
where (days_b_screening_arrest <= 30) and 
      (days_b_screening_arrest >= -30) and
      (is_recid != -1) and
      (c_charge_degree != 'O');

     