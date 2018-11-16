cd "C:\Users\Marcos Lee\Dropbox\Rice\Courses\Labor\KW94\"

use keane_wolpin_1994_restat.dta, clear

gen school_c = 0
replace school_c = 1 if choice == 3

gen work1 = 0
replace work1 = 1 if choice == 1

gen work2 = 0
replace work2 = 1 if choice == 2

gen home = 0
replace home = 1 if choice == 4


bysort period: egen avgs = mean(school_c)
bysort period: egen avgw1 = mean(work1)
bysort period: egen avgw2 = mean(work2)
bysort period: egen avgh = mean(home)


tabstat avgw1, by(period)
tabstat avgw2, by(period)
tabstat avgs, by(period)
tabstat avgh, by(period)


foreach i in 1 2 3{
	import delimited df`i'.csv, clear
	
	bysort period: egen avgs = mean(school_c)
	bysort period: egen avgw1 = mean(work1)
	bysort period: egen avgw2 = mean(work2)
	bysort period: egen avgh = mean(home)


	tabstat avgw1, by(period)
	tabstat avgw2, by(period)
	tabstat avgs, by(period)
	tabstat avgh, by(period)
}
