
select  count(*)
 from
    store_sales
   ,date_dim       d1
   ,item
   ,store
 where
    d1.d_year = 1999 
 and d1.d_date_sk = ss_sold_date_sk
 and i_item_sk  = ss_item_sk 
 and s_store_sk  = ss_store_sk
 and s_state in ('TN','AL','SD','TN',
                 'SD','SD','SD','SD');


