
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (2001) and
        ((i_category in ('Jewelry','Men','Women') and
          i_class in ('womens watch','accessories','fragrances')
         )
      or (i_category in ('Children','Home','Electronics') and
          i_class in ('toddlers','blinds/shades','personal') 
        ));


