
select  count(*)
       from store_sales
           ,date_dim
           ,store
           ,household_demographics
           ,customer_address 
       where store_sales.ss_sold_date_sk = date_dim.d_date_sk
         and store_sales.ss_store_sk = store.s_store_sk  
        and store_sales.ss_hdemo_sk = household_demographics.hd_demo_sk
        and store_sales.ss_addr_sk = customer_address.ca_address_sk
        and date_dim.d_dom between 1 and 2 
        and (household_demographics.hd_dep_count = 4 or
             household_demographics.hd_vehicle_count= 3)
        and date_dim.d_year in (1998,1998+1,1998+2)
        and store.s_city in ('Fairview','Pleasant Hill');


