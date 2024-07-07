
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (2000) and
        ((i_category in ('Electronics','Sports','Music') and
          i_class in ('audio','sailing','pop')
         )
      or (i_category in ('Shoes','Men','Books') and
          i_class in ('kids','shirts','computers') 
        ));


