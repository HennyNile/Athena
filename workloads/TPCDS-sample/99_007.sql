
select  count(*)
from
   catalog_sales
  ,warehouse
  ,ship_mode
  ,call_center
  ,date_dim
where
    d_month_seq between 1201 and 1201 + 11
and cs_ship_date_sk   = d_date_sk
and cs_warehouse_sk   = w_warehouse_sk
and cs_ship_mode_sk   = sm_ship_mode_sk
and cs_call_center_sk = cc_call_center_sk;


