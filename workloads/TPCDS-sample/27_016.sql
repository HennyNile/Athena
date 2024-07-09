
select  count(*)
 from store_sales, customer_demographics, date_dim, store, item
 where ss_sold_date_sk = d_date_sk and
       ss_item_sk = i_item_sk and
       ss_store_sk = s_store_sk and
       ss_cdemo_sk = cd_demo_sk and
       cd_gender = 'M' and
       cd_marital_status = 'D' and
       cd_education_status = 'Advanced Degree' and
       d_year = 2001 and
       s_state in ('TN','SD', 'AL', 'SD', 'SD', 'SD');


