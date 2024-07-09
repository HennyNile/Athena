
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (1999) and
        ((i_category in ('Children','Men','Electronics') and
          i_class in ('toddlers','pants','memory')
         )
      or (i_category in ('Sports','Jewelry','Shoes') and
          i_class in ('basketball','costume','mens') 
        ));


