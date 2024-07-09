
select  count(*)
 from
    web_sales
   ,date_dim       d1
   ,item
 where
    d1.d_month_seq between 1184 and 1184+11
 and d1.d_date_sk = ws_sold_date_sk
 and i_item_sk  = ws_item_sk;


