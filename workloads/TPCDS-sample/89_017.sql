
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (2001) and
        ((i_category in ('Music','Men','Books') and
          i_class in ('pop','pants','travel')
         )
      or (i_category in ('Home','Shoes','Women') and
          i_class in ('paint','kids','maternity') 
        ));


