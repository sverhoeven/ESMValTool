; #############################################################################
; ESMValTool CMORizer for CALIPSO-Lidar Level 3 Ice Cloud Data, Standard V1-00
; #############################################################################
;
; Tier
;    Tier 2: other freely-available dataset.
;
; Source
;    EarthData via https://eosweb.larc.nasa.gov/project/calipso/
;            cal_lid_l3_ice_cloud_standard_v1_00
;
; Last access
;    20200408
;
; Download and processing instructions
; (requires EarthData login; see https://urs.earthdata.nasa.gov/)
;    1) Go to https://eosweb.larc.nasa.gov/project/calipso/
;                    cal_lid_l3_ice_cloud_standard_v1_00
;    2) Click on "Order data"
;    3) Select Granules containing both (day and night)
;    4) Download selected granules
;    5) Enter EarthData login and password (https://urs.earthdata.nasa.gov/)
;    6) Follow download instructions in email from EarthData and put all
;       files in the same directory
;
; Modification history
;    20200414-laue_axel: written
;
; #############################################################################
loadscript(getenv("esmvaltool_root") + \
           "/esmvaltool/cmorizers/obs/interface.ncl")

begin

  ; Script name (for logger)
  DIAG_SCRIPT = "cmorize_obs_calipso-icecloud.ncl"

  ; Source name
  OBSNAME = "CALIPSO-ICECLOUD"

  ; Tier
  TIER = 2

  ; Period
  YEAR1 = 2007
  YEAR2 = 2007;2016

  ; Selected variable (standard name)
  VAR = (/"cli"/)

  ; MIP
  MIP = (/"AMon"/)

  ; Frequency
  FREQ = (/"mon"/)

  ; CMOR table
  CMOR_TABLE = getenv("cmor_tables") + \
    (/"/cmip5/Tables/CMIP5_cfMon"/)

  ; Type
  TYPE = "sat"

  ; Version
  VERSION = "1-00"

  ; Global attributes
  SOURCE = "https://eosweb.larc.nasa.gov/project/calipso/" + \
    "cal_lid_l3_ice_cloud_standard_v1_00"
  REF = "doi:10.5067/CALIOP/CALIPSO/L3_Ice_Cloud-Standard-V1-00"
  COMMENT = "Atmospheric Science Data Center (ASDC)"

end

begin

  vv = 0
  log_info("Processing " + VAR(vv) + " (" + MIP(vv) + ")")

  time = create_timec(YEAR1, YEAR2)
  date = cd_calendar(time, 1)

  ; Create timeseries
  do yy = YEAR1, YEAR2

    syear = sprinti("%i", yy)
    do mm = 1, 12

      smonth = sprinti("%0.2i", mm)

      print(syear + "-" + smonth)

      ; Read data

      ; There are 3 granules available:
      ;   day:   CAL_LID_L3_Ice_Cloud-Standard-V1-00.yyyy-mmD.hdf
      ;   night: CAL_LID_L3_Ice_Cloud-Standard-V1-00.yyyy-mmN.hdf
      ;   both:  CAL_LID_L3_Ice_Cloud-Standard-V1-00.yyyy-mmA.hdf
      ; Here, we only use the granule "A" (= day + night).

      fname = input_dir_path + "CAL_LID_L3_Ice_Cloud-Standard-V1-00." + \
        syear + "-" + smonth + "A.hdf"

      ; No input file found
      if (ismissing(fname)) then
        error_msg("f", DIAG_SCRIPT, "main", "input file for " + syear + \
          "-" + smonth + " not found")
      end if

      ; Extract data
      f = addfile(fname, "r")

      iwc_hist = f->Ice_Water_Content_Histogram
      iwc_bnds = f->Ice_Water_Content_Bin_Boundaries
      cldfree_samp = f->Cloud_Free_Samples
      cld_samp = f->Cloud_Samples

      ; Create output array
      if (.not.isdefined("output")) then
        alt = round(f->Altitude_Midpoint * 1000.0, 0)  ; km, round to 1 m
        lat = f->Latitude_Midpoint
        lon = f->Longitude_Midpoint
        ndims = new(4, integer)
        ndims(0) = dimsizes(time)
        ndims(1) = dimsizes(alt)
        ndims(2) = dimsizes(lat)
        ndims(3) = dimsizes(lon)
        output = new(ndims, float)
        output!0 = "time"
        output&time = time
        output!1 = "alt"
        output&alt = alt
        output!2 = "lat"
        output&lat = lat
        output!3 = "lon"
        output&lon = lon
      end if

      ; Calculate grid-box average ice water content from ice water content
      ; histogram. Formula from https://www-calipso.larc.nasa.gov/resources/
      ;                calipso_users_guide/qs/cal_lid_l3_ice_cloud_v1-00.php
      ; (section "In-cloud IWC and Grid-averaged IWC").
      ;
      ; Note: 1) ice water content is derived from extiction coefficient. The
      ;       extinction coefficient can be negative, thus negative ice water
      ;       contents are produced. These are included in the level 3 product
      ;       used here to "accurately represent the retrieved population in
      ;       the level 2 data product" (that was used to derive the level 3
      ;       product used here).
      ;       However, these unphysical ice water contents are excluded here
      ;       when calculating the grid box average ice water content for
      ;       comparison with earth system models.
      ;       2) ice water content is only calculated for clouds with
      ;       1.0e-5 <= iwc <= 1.0 g/m3.

      ; find first bin with iwc >= 1e-5 g/m3 (first bin with valid data)

      tmp = ind(iwc_bnds(:, 0) .ge. 1.0e-5)  ; min bound
      if (all(ismissing(tmp))) then
        error_msg("f", DIAG_SCRIPT, "main", "invalid bounds for histogram")
      end if
      i0 = tmp(0)
      delete(tmp)

      ; find last bin with iwc <= 1.0 g/m3 (last bin with valid data)
      tmp = ind(iwc_bnds(:, 2) .le. 1.0)  ; max bound
      if (all(ismissing(tmp))) then
        error_msg("f", DIAG_SCRIPT, "main", "invalid bounds for histogram")
      end if
      i1 = tmp(dimsizes(tmp) - 1)
      delete(tmp)

      iwc = iwc_hist * conform_dims(dimsizes(iwc_hist), iwc_bnds(:, 1), 3)
      samples = tofloat(cldfree_samp + cld_samp)
      samples@_FillValue = -999.
      samples = where(samples.gt.0., samples, samples@_FillValue)
      iwc_avg = dim_sum(iwc(:, :, :, i0:i1)) / samples

      iwc_avg!0 = "lat"
      iwc_avg!1 = "lon"
      iwc_avg!2 = "alt"

      output(ind(toint(yy * 100 + mm).eq.date), :, :, :) = \
        iwc_avg(alt|:, lat|:, lon|:)

      delete(fname)
      delete(f)
      delete(iwc_avg)
      delete(iwc)
      delete(samples)
    end do
  end do

  ; Set fill value
  output = where(output.eq.-999, output@_FillValue, output)

  ; Format coordinates
  output!0 = "time"
  output!1 = "alt"
  output!2 = "lat"
  output!3 = "lon"
  format_coords(output, YEAR1 + "0101", YEAR2 + "1231", FREQ(vv))

  ; Set variable attributes
  tmp = format_variable(output, VAR(vv), CMOR_TABLE(vv))
  delete(output)
  output = tmp
  delete(tmp)

  ; Calculate coordinate bounds
  bounds = guess_coord_bounds(output, FREQ(vv))

  ; Set global attributes
  gAtt = set_global_atts(OBSNAME, TIER, SOURCE, REF, COMMENT)

  ; Output file
  DATESTR = YEAR1 + "01-" + YEAR2 + "12"
  fout = output_dir_path + \
    str_join((/"OBS", OBSNAME, TYPE, VERSION, \
               MIP(vv), VAR(vv), DATESTR/), "_") + ".nc"

  ; Write variable
  write_nc(fout, VAR(vv), output, bounds, gAtt)
  delete(gAtt)
  delete(output)
  delete(bounds)

end
