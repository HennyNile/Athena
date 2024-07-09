
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (2001) and
        ((i_category in ('Sports','Women','Home') and
          i_class in ('guns','fragrances','curtains/drapes')
         )
      or (i_category in ('Electronics','Shoes','Jewelry') and
          i_class in ('audio','womens','gold') 
        ));


