
select  count(*)
from item, store_sales, date_dim, store
where ss_item_sk = i_item_sk and
      ss_sold_date_sk = d_date_sk and
      ss_store_sk = s_store_sk and
      d_year in (1998) and
        ((i_category in ('Sports','Music','Jewelry') and
          i_class in ('fitness','country','mens watch')
         )
      or (i_category in ('Electronics','Books','Home') and
          i_class in ('disk drives','home repair','mattresses') 
        ));


