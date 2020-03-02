#!MC 1410
$!VARSET |MFBD| = '/'

$!ALTERDATA
  EQUATION = '{r [R]} = sqrt(V1**2 + V2**2 + V3**2)'
$!VARSET |rindex| = |numvars|

$!GLOBALTHREEDVECTOR UVAR = 8
$!GLOBALTHREEDVECTOR VVAR = 9
$!GLOBALTHREEDVECTOR WVAR = 10

$!STREAMTRACELAYERS SHOW = YES
$!STREAMTRACE RESETDELTATIME
$!STREAMATTRIBUTES RODRIBBON{WIDTH = 32.9090909090908923}

$!VarSet |rPoint|        = 2.5

                # =============================================================
                # Function to create a streamline
                # Inputs |1| -> rPoint
                #        |2| -> Latitude
                #        |3| -> Phi

                $!MACROFUNCTION NAME = "create_streamzone"

                    $!VarSet |x_set| = (|1|*sin(|2|)*cos(|3|))
                    $!VarSet |y_set| = (|1|*sin(|2|)*sin(|3|))
                    $!VarSet |z_set| = (|1|*cos(|2|))

                    $!STREAMTRACE ADD
                        STREAMTYPE = VOLUMELINE
                        STREAMDIRECTION = BOTH
                        STARTPOS
                            {
                            X = |x_set|
                            Y = |y_set|
                            Z = |z_set|
                            }
                    $!CREATESTREAMZONES
                    $!STREAMTRACE DELETEALL

                $!ENDMACROFUNCTION

                # -------------------------------------------------------------
                # Function to check if a streamzone is open or closed
                # Inputs |1| -> Streamzone number / ID
                #        |2| -> rPoint

                $!MACROFUNCTION NAME = "check_open_closed"
                    $!ActiveFieldMaps = [|1|]
                    $!GETFIELDVALUE |StartPoint|
                        Zone = |1|
                        Var = |rindex|
                        Index = 1

                    $!GETFIELDVALUE |EndPoint|
                        Zone = |1|
                        Var = |rindex|
                        Index = |MaxI|

                    $!IF |StartPoint| > |2|
                        $!VarSet |Closed| = 0
                    $!ELSEIF |EndPoint| > |2|
                        $!VarSet |Closed| = 0
                    $!ELSE
                        $!VarSet |Closed| = 1
                    $!ENDIF

                $!ENDMACROFUNCTION


		
# ============================================================================
# Create the dayside magnetopause zone

$!VARSET |ntheta| = 30
$!VARSET |nphi|   = 30

$!CREATERECTANGULARZONE
IMAX     = |ntheta|
JMAX     = |nphi|
X1       = 0.5235
Y1       = (-3.1415 / 2)
X2       = 2.6179
Y2       = (3.1415 / 2)

$!ALTERDATA [|numzones|]
	EQUATION = '{Theta [rad]} = V1'
$!ALTERDATA [|numzones|]
	EQUATION = '{Phi [rad]} = V2'
$!ALTERDATA [|numzones|]
	EQUATION = '{X [R]} = sin({Theta [rad]}) * cos({Phi [rad]})'
$!ALTERDATA [|numzones|]
	EQUATION = '{Y [R]} = sin({Theta [rad]}) * sin({Phi [rad]})'
$!ALTERDATA [|numzones|]
	EQUATION = '{Z [R]} = cos({Theta [rad]})'

$!VARSET |thetaindex| = (|numvars|-1)
$!VARSET |phiindex| = |numvars|



$!LOOP |ntheta|
	$!VARSET |i| = |loop|
	$!LOOP |nphi|
		$!VARSET |j| = |loop|

		$!VARSET |index| = (|i| + ((|j|-1)*|ntheta|))
		# Get the theta and phi values for this (i,j)
		
		$!GETFIELDVALUE |theta|
			ZONE = |numzones|
			VAR   = |thetaindex|
			INDEX = |index|

		$!GETFIELDVALUE |phi|
			ZONE = |numzones|
			VAR   = |phiindex|
			INDEX = |index|
		
		$!ActiveFieldMaps = [1]
		# Find the magnetopause location in R for this (i,j)
		$!VarSet |Outer| = 30.0
		$!VarSet |Inner| = 2.0

		$!WHILE (|Outer| - |Inner|) > 0.1
			$!VarSet |Centre| = ((|Outer| + |Inner|) / 2)
			$!RUNMACROFUNCTION "create_streamzone" (|Centre|, |theta|, |phi|)
			$!RUNMACROFUNCTION "check_open_closed" (|numZones|, |rPoint|)
			$!ActiveFieldMaps = [1]
			$!DELETEZONES [|numZones|]
                        	
                    	$!IF |Closed| == 1
                            $!VarSet |Inner| = |Centre|
                        $!ELSE
                            $!VarSet |Outer| = |Centre|
                        $!ENDIF

		$!ENDWHILE
		

		$!VARSET |x| = (|Outer| * sin(|theta|) * cos(|phi|))
		$!VARSET |y| = (|Outer| * sin(|theta|) * sin(|phi|))
		$!VARSET |z| = (|Outer| * cos(|theta|))

		$!ActiveFieldMaps = [|numzones|]
		$!SETFIELDVALUE 
		Zone = |numzones|
                Var = 1
                Index = |index|
                FieldValue = |x|
                AutoBranch = YES
		
		$!SETFIELDVALUE 
		Zone = |numzones|
                Var = 2
                Index = |index|
                FieldValue = |y|
                AutoBranch = YES
		
		$!SETFIELDVALUE 
		Zone = |numzones|
                Var = 3
                Index = |index|
                FieldValue = |z|
                AutoBranch = YES
		

	$!ENDLOOP
$!ENDLOOP	

# ===================================================================================================
# Interpolate field data and calculate normal energy flux on magnetopause zone

# Interpolate data
$!INVERSEDISTINTERPOLATE
	SOURCEZONES = [1]
	DESTINATIONZONE = 2
	VARLIST = [1-|numvars|]


# Create MP surface normal vectors using built in function
$!EXTENDEDCOMMAND
	  COMMANDPROCESSORID = 'CFDAnalyzer3'
	  COMMAND = 'CALCULATE FUNCTION = GRIDKUNITNORMAL VALUELOCATION = CELLCENTERED'


# Calculate energy flux for all zones
$!ALTERDATA[|numzones|] #Electric Field (no resistivity)
	EQUATION = '{E_x [mV/km]} = ({U_z [km/s]}*{B_y [nT]} - {U_y [km/s]}*{B_z [nT]})'

$!ALTERDATA[|numzones|]
        EQUATION = '{E_y [mV/km]} = ({U_x [km/s]}*{B_z [nT]} - {U_z [km/s]}*{B_x [nT]})'

$!ALTERDATA[|numzones|]
        EQUATION = '{E_z [mV/km]} = ({U_y [km/s]}*{B_x [nT]} - {U_x [km/s]}*{B_y [nT]})'



$!ALTERDATA[|numzones|] #Poynting Flux
        EQUATION = '{ExB_x [kW/km^2]} = -(1/1.25663706)*({E_z [mV/km]}*{B_y [nT]} - {E_y [mV/km]}*{B_z [nT]})*1e-6'

$!ALTERDATA[|numzones|]
        EQUATION = '{ExB_y [kW/km^2]} = -(1/1.25663706)*({E_x [mV/km]}*{B_z [nT]} - {E_z [mV/km]}*{B_x [nT]})*1e-6'

$!ALTERDATA[|numzones|]
        EQUATION = '{ExB_z [kW/km^2]} = -(1/1.25663706)*({E_y [mV/km]}*{B_x [nT]} - {E_x [mV/km]}*{B_y [nT]})*1e-6'



$!ALTERDATA[|numzones|] #Total Energy Flux
        EQUATION = '{K_x [kW/km^2]} = 1e-6*(1000*{P [nPa]}*(1.666667/(1.666667-1)) + 1e-3*{Rho [amu/cm^3]}/2*({U_x [km/s]}**2+{U_y [km/s]}**2+{U_z [km/s]}**2))*{U_x [km/s]}  +  {ExB_x [kW/km^2]}'

$!ALTERDATA[|numzones|]
        EQUATION = '{K_y [kW/km^2]} = 1e-6*(1000*{P [nPa]}*(1.666667/(1.666667-1)) + 1e-3*{Rho [amu/cm^3]}/2*({U_x [km/s]}**2+{U_y [km/s]}**2+{U_z [km/s]}**2))*{U_y [km/s]}  +  {ExB_y [kW/km^2]}'

$!ALTERDATA[|numzones|]
        EQUATION = '{K_z [kW/km^2]} = 1e-6*(1000*{P [nPa]}*(1.666667/(1.666667-1)) + 1e-3*{Rho [amu/cm^3]}/2*({U_x [km/s]}**2+{U_y [km/s]}**2+{U_z [km/s]}**2))*{U_z [km/s]}  +  {ExB_z [kW/km^2]}'
		



# Calculate othogonal projection of energy flux through magnetopause
$!ALTERDATA[|numzones|] #Component Normal Flux
        EQUATION = '{Kn_x [kW/km^2]} = ({K_x [kW/km^2]}*{X Grid K Unit Normal} + {K_y [kW/km^2]}*{Y Grid K Unit Normal} + {K_z [kW/km^2]}*{Z Grid K Unit Normal}) / sqrt({X Grid K Unit Normal}**2 + {Y Grid K Unit Normal}**2 + {Z Grid K Unit Normal}**2) * {X Grid K Unit Normal}'

$!ALTERDATA[|numzones|]
        EQUATION = '{Kn_y [kW/km^2]} = ({K_x [kW/km^2]}*{X Grid K Unit Normal} + {K_y [kW/km^2]}*{Y Grid K Unit Normal} + {K_z [kW/km^2]}*{Z Grid K Unit Normal}) / sqrt({X Grid K Unit Normal}**2 + {Y Grid K Unit Normal}**2 + {Z Grid K Unit Normal}**2) * {Y Grid K Unit Normal}'

$!ALTERDATA[|numzones|]
        EQUATION = '{Kn_z [kW/km^2]} = ({K_x [kW/km^2]}*{X Grid K Unit Normal} + {K_y [kW/km^2]}*{Y Grid K Unit Normal} + {K_z [kW/km^2]}*{Z Grid K Unit Normal}) / sqrt({X Grid K Unit Normal}**2 + {Y Grid K Unit Normal}**2 + {Z Grid K Unit Normal}**2) * {Z Grid K Unit Normal}'


$!ALTERDATA[|numzones|] #Magnitude Normal Flux
        EQUATION = '{K_in [Kw/km^2]} = ({Kn_x [kW/km^2]}*{X Grid K Unit Normal} + {Kn_y [kW/km^2]}*{Y Grid K Unit Normal} + {Kn_z [kW/km^2]}*{Z Grid K Unit Normal}) / sqrt({X Grid K Unit Normal}**2 + {Y Grid K Unit Normal}**2 + {Z Grid K Unit Normal}**2)'


