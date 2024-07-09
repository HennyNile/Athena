
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (2001) and
        ((i_category in ('Men','Shoes','Music') and
          i_class in ('shirts','womens','country')
         )
      or (i_category in ('Jewelry','Children','Books') and
          i_class in ('semi-precious','school-uniforms','cooking') 
        ));


