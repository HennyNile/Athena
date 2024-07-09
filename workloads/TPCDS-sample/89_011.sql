
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (2000) and
        ((i_category in ('Jewelry','Children','Books') and
          i_class in ('birdal','toddlers','sports')
         )
      or (i_category in ('Sports','Women','Home') and
          i_class in ('archery','swimwear','bathroom') 
        ));


