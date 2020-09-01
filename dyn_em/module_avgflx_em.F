!WRF:MODEL_LAYER:DYNAMICS
!
MODULE module_avgflx_em

  USE module_bc
  USE module_model_constants
  USE module_wrf_error
  USE module_state_description

CONTAINS

!-------------------------------------------------------------------------------


  subroutine zero_avgflx(avgflx_rum,avgflx_rvm,avgflx_wwm, &
       & ids, ide, jds, jde, kds, kde,           &
       & ims, ime, jms, jme, kms, kme,           &
       & its, ite, jts, jte, kts, kte, do_cu, do_avgflx_em, output_avgfluxes,    &
       & avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1, &
       & hist_mean, hist_mean_8w)  
    IMPLICIT NONE

    INTEGER , INTENT(IN)        :: ids, ide, jds, jde, kds, kde,  &
         ims, ime, jms, jme, kms, kme,  &
         its, ite, jts, jte, kts, kte
    INTEGER , INTENT(IN)        :: do_avgflx_em, output_avgfluxes
    LOGICAL, INTENT(IN) :: do_cu

    REAL,     DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_rum,avgflx_rvm,avgflx_wwm

    REAL , OPTIONAL, DIMENSION( ims:ime, kms:kme, jms:jme, num_hist_mean), INTENT(INOUT) :: hist_mean
    REAL , OPTIONAL, DIMENSION( ims:ime, kms:kme, jms:jme, num_hist_mean_8w), INTENT(INOUT) :: hist_mean_8w

    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1

    INTEGER :: i,j,k

    if (output_avgfluxes .eq. 1) then
      DO j=jts,jte
         DO k=kts,kte
            DO i=its,ite
               hist_mean(i,k,j,:) = 0.
               hist_mean_8w(i,k,j,:) = 0.
            end DO
         end DO
       end DO
    endif

    if (do_avgflx_em .eq. 1) then
    DO j=jts,jte
       DO k=kts,kte
          DO i=its,ite
             avgflx_rum(i,k,j) = 0.
             avgflx_rvm(i,k,j) = 0.
             avgflx_wwm(i,k,j) = 0.
          end DO
       end DO
    end DO
    endif

    if (do_cu .and. &
         & present(avgflx_cfu1) .and. present(avgflx_cfd1) .and. present(avgflx_dfu1) &
         & .and. present(avgflx_efu1) .and. present(avgflx_dfd1) .and. present(avgflx_efd1) ) then
       DO j=jts,jte
          DO k=kts,kte
             DO i=its,ite
                avgflx_cfu1(i,k,j) = 0.
                avgflx_cfd1(i,k,j) = 0.
                avgflx_dfu1(i,k,j) = 0.
                avgflx_efu1(i,k,j) = 0.
                avgflx_dfd1(i,k,j) = 0.
                avgflx_efd1(i,k,j) = 0.
             end DO
          end DO
       end DO
    end if

    return
  end subroutine zero_avgflx

  subroutine upd_avgflx(avgflx_count,avgflx_rum,avgflx_rvm,avgflx_wwm, &
       &   ru_m, rv_m, ww_m, &
       & ids, ide, jds, jde, kds, kde,           &
       & ims, ime, jms, jme, kms, kme,           &
       & its, ite, jts, jte, kts, kte, do_cu,    &
       & cfu1,cfd1,dfu1,efu1,dfd1,efd1,          &
       & avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1,   &
       & config_flags,                                                              &
       & sgs_fluxes, sgs_fluxes_8w, nba_mij, hist_mean, hist_mean_8w,               &
       & ruth_tot, rvth_tot, wwth_tot, ruq_tot, rvq_tot, wwq_tot,                   &
       & t_tend_radlw, t_tend_radsw, t_tend_mp, qv_tend_mp,                         &
       & Ustag, Vstag, Wstag, ru, rv, ww, thm, Q,                                   &
       & ph_1, ph_2, phb, zx, zy,                                                   &
       & dnw, fnm, fnp, mut, c1h, c2h, c1f, c2f, cf1, cf2, cf3,                     &
       & last_timestep, dt                                                          )
    

    IMPLICIT NONE

    INTEGER , INTENT(IN)        :: ids, ide, jds, jde, kds, kde,  &
         ims, ime, jms, jme, kms, kme,  &
         its, ite, jts, jte, kts, kte

    INTEGER , INTENT(IN)        :: avgflx_count
    LOGICAL, INTENT(IN) :: do_cu, last_timestep
    REAL, INTENT(IN) :: dt, cf1, cf2, cf3

    REAL, DIMENSION(ims:ime, kms:kme, jms:jme) , INTENT(IN) :: &
         ru_m, rv_m, ww_m,             & 
         Ustag, Vstag, Wstag, ru, rv, ww,                          &
         thm, Q, ph_1, ph_2, phb, zx, zy,                     &
         ruth_tot, rvth_tot, wwth_tot, ruq_tot, rvq_tot, wwq_tot,  &
         t_tend_radlw, t_tend_radsw, t_tend_mp, qv_tend_mp

    REAL , DIMENSION( ims:ime, kms:kme, jms:jme, num_sgs_fluxes), INTENT(IN) :: sgs_fluxes
    REAL , DIMENSION( ims:ime, kms:kme, jms:jme, num_sgs_fluxes_8w), INTENT(IN) :: sgs_fluxes_8w
    REAL , DIMENSION( ims:ime, kms:kme, jms:jme, num_nba_mij), INTENT(IN) :: nba_mij
    REAL , DIMENSION( kms:kme ) ,    INTENT(IN   ) :: dnw, fnm, fnp, c1h, c2h, c1f, c2f
    REAL , DIMENSION( ims:ime, jms:jme ), INTENT(IN   ) :: mut



    TYPE(grid_config_rec_type), INTENT(IN   ) :: config_flags


    REAL,     DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_rum,avgflx_rvm,avgflx_wwm

    REAL , DIMENSION( ims:ime, kms:kme, jms:jme, num_hist_mean), INTENT(INOUT) :: hist_mean
    REAL , DIMENSION( ims:ime, kms:kme, jms:jme, num_hist_mean_8w), INTENT(INOUT) :: hist_mean_8w


    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(IN) ::    &
         cfu1,cfd1,dfu1,efu1,dfd1,efd1
    REAL,     OPTIONAL, DIMENSION( ims:ime , kms:kme , jms:jme ) , INTENT(INOUT) ::    &
         avgflx_cfu1,avgflx_cfd1,avgflx_dfu1,avgflx_efu1,avgflx_dfd1,avgflx_efd1

    REAL,    DIMENSION( its-1:ite , kts:kte , jts-1:jte ) ::    rho_d, th, thd

    INTEGER :: i,j,k,i_end,j_end
    REAL :: local_count, rho_8w, rhox, rhoy, rhoz, rho_mean, rhox_mean, rhoy_mean, rhoz_mean, &
            thetax, thetay, thetaz, qx, qy, qz, qfac,        &
            zx_8w, zy_8w, u_8w, v_8w, dzdt,              &
            corr_uth, corr_vth, corr_dthdt, corr_uq, corr_vq, corr_dqdt, &
            wdiag, muz, mu, dphidn
    REAL :: thetax2, thetay2, thetaz2, qx2, qy2, qz2

    i_end = min(ite,ide-1)
    j_end = min(jte,jde-1)
    local_count = real(avgflx_count)


    if (config_flags%output_avgfluxes .eq. 1) then

    DO j=jts-1,j_end
    DO k=kts,kte-1
    DO i=its-1,i_end
      !calculate dry density
      mu = c1h(k)*mut(i,j)+c2h(k)
      dphidn = ((phb(i,k+1,j)+ph_2(i,k+1,j)) - (phb(i,k,j)+ph_2(i,k,j)))/dnw(k)
      rho_d(i,k,j) = mu/(-dphidn)
      hist_mean(i,k,j,P_RHOD_MEAN) = (local_count*hist_mean(i,k,j,P_RHOD_MEAN)  + rho_d(i,k,j)  )/(local_count+1.)
      IF ( config_flags%use_theta_m .EQ. 1 ) THEN
        thd(i,k,j) = (thm(i,k,j) + t0)/(1 + rvovrd*Q(i,k,j)) - t0
        IF ( config_flags%output_dry_theta_fluxes .eq. 1) THEN
          th(i,k,j) = thd(i,k,j)
        ELSE
          th(i,k,j) = thm(i,k,j)
        ENDIF
      ELSE
        thd(i,k,j) = thm(i,k,j)
        th(i,k,j) = thm(i,k,j)
      ENDIF
    ENDDO
    ENDDO
    ENDDO

    DO j=jts,j_end
      DO k=kts,kte-1
        DO i=its,i_end

          hist_mean_8w(i,k,j,P_Z_MEAN) = (local_count*hist_mean_8w(i,k,j,P_Z_MEAN)  + (phb(i,k,j)+ph_2(i,k,j))/g  )/(local_count+1.)

          IF (config_flags%output_res_fluxes==1) THEN
            !stagger density
            if ( k .gt. kts) then
              rho_8w = fnm(k)*rho_d(i,k,j)+ fnp(k)*rho_d(i,k-1,j)
            else
              rho_8w = cf1*rho_d(i,1,j)+cf2*rho_d(i,2,j)+cf3*rho_d(i,3,j)
            endif
            if (config_flags%hesselberg_avg) then
              rhox = 0.5*(rho_d(i,k,j)+rho_d(i-1,k,j))
              rhoy = 0.5*(rho_d(i,k,j)+rho_d(i,k,j-1))
              rhoz = rho_8w
            else
              rhox = 1
              rhoy = 1
              rhoz = 1
            endif

            !determine staggered theta from total fluxes
            thetax = ruth_tot(i,k,j)/ru(i,k,j)
            thetay = rvth_tot(i,k,j)/rv(i,k,j)
            if ( k .gt. kts) then
              thetaz = wwth_tot(i,k,j)/ww(i,k,j)
            else
              thetaz = cf1*th(i,1,j)+cf2*th(i,2,j)+cf3*th(i,3,j)
            endif
            !2nd order staggering
            thetax2 = 0.5*(th(i-1,k,j) + th(i,k,j))
            thetay2 = 0.5*(th(i,k,j-1) + th(i,k,j))
            if ( k .gt. kts) then
              thetaz2 = (fnp(k)* th(i,k-1,j) + fnm(k)* th(i,k,j))
            ELSE
              thetaz2 = thetaz
            endif

            !staggered q, different vertical velocity needed
            qx = ruq_tot(i,k,j)/ru_m(i,k,j)
            qy = rvq_tot(i,k,j)/rv_m(i,k,j)
            if ( k .gt. kts) then
              qz = wwq_tot(i,k,j)/ww_m(i,k,j)
            else
              qz = cf1*Q(i,1,j)+cf2*Q(i,2,j)+cf3*Q(i,3,j)
            endif

            !2nd order staggering

            qx2 = 0.5*(Q(i,k,j)+Q(i-1,k,j))
            qy2 = 0.5*(Q(i,k,j)+Q(i,k,j-1))
            if ( k .gt. kts) then
              qz2 = (fnm(k)*Q(i,k,j) + fnp(k)*Q(i,k-1,j))
            else
              qz2 = qz
            endif

            !compute cartesian corrections
            zx_8w = 0.5*(zx(i,k,j)+zx(i+1,k,j))
            zy_8w = 0.5*(zy(i,k,j)+zy(i,k,j+1))
            dzdt = (ph_2(i,k,j) - ph_1(i,k,j))/(g*dt)
            if ( k .gt. kts) then
              u_8w = fnm(k)*0.5*(Ustag(i,k,j)+Ustag(i+1,k,j)) + fnp(k)*0.5*(Ustag(i,k-1,j)+Ustag(i+1,k-1,j))
              v_8w = fnm(k)*0.5*(Vstag(i,k,j)+Vstag(i,k,j+1)) + fnp(k)*0.5*(Vstag(i,k-1,j)+Vstag(i,k-1,j+1))
            else
              u_8w = cf1*0.5*(Ustag(i,1,j)+Ustag(i+1,1,j)) + cf2*0.5*(Ustag(i,2,j)+Ustag(i+1,2,j)) + cf3*0.5*(Ustag(i,3,j)+Ustag(i+1,3,j))
              v_8w = cf1*0.5*(Vstag(i,1,j)+Vstag(i,1,j+1)) + cf2*0.5*(Vstag(i,2,j)+Vstag(i,2,j+1)) + cf3*0.5*(Vstag(i,3,j)+Vstag(i,3,j+1))
            endif

            corr_uth = rho_8w*zx_8w*u_8w*thetaz
            corr_vth = rho_8w*zy_8w*v_8w*thetaz
            corr_dthdt = rho_8w*dzdt*thetaz

            corr_uq = rho_8w*zx_8w*u_8w*qz
            corr_vq = rho_8w*zy_8w*v_8w*qz
            corr_dqdt = rho_8w*dzdt*qz

            wdiag = dzdt + zx_8w*u_8w + zy_8w*v_8w + ww(i,k,j)/(-rho_8w*g)
            muz = c1f(k)*mut(i,j)+c2f(k)


            if (last_timestep .and. config_flags%hesselberg_avg ) then
              !normalize means by density
              rho_mean = hist_mean(i,k,j,P_RHOD_MEAN)
              rhox_mean = 0.5*(hist_mean(i,k,j,P_RHOD_MEAN) + hist_mean(i-1,k,j,P_RHOD_MEAN))
              rhoy_mean = 0.5*(hist_mean(i,k,j,P_RHOD_MEAN) + hist_mean(i,k,j-1,P_RHOD_MEAN))
              if ( k .gt. kts) then
                rhoz_mean = fnm(k)*hist_mean(i,k,j,P_RHOD_MEAN) + fnp(k)*hist_mean(i,k-1,j,P_RHOD_MEAN)
              else
                rhoz_mean = cf1*hist_mean(i,1,j,P_RHOD_MEAN) + cf2*hist_mean(i,2,j,P_RHOD_MEAN) + cf3*hist_mean(i,3,j,P_RHOD_MEAN)
              endif
            ELSE
              rho_mean  = 1
              rhox_mean = 1
              rhoy_mean = 1
              rhoz_mean = 1
            ENDIF

            !mean variables
            hist_mean(i,k,j,P_U_MEAN) = (local_count*hist_mean(i,k,j,P_U_MEAN) + Ustag(i,k,j)*rhox  )/(local_count+1.)/rhox_mean
            hist_mean(i,k,j,P_V_MEAN) = (local_count*hist_mean(i,k,j,P_V_MEAN) + Vstag(i,k,j)*rhoy  )/(local_count+1.)/rhoy_mean
            hist_mean_8w(i,k,j,P_W_MEAN) = (local_count*hist_mean_8w(i,k,j,P_W_MEAN) + Wstag(i,k,j)*rhoz  )/(local_count+1.)/rhoz_mean
            hist_mean_8w(i,k,j,P_OM_MEAN) = (local_count*hist_mean_8w(i,k,j,P_OM_MEAN) + ww(i,k,j)/muz*rhoz )/(local_count+1.)/rhoz_mean
            hist_mean_8w(i,k,j,P_WD_MEAN) = (local_count*hist_mean_8w(i,k,j,P_WD_MEAN) + wdiag*rhoz )/(local_count+1.)/rhoz_mean

            hist_mean(i,k,j,P_TH_MEAN) = (local_count*hist_mean(i,k,j,P_TH_MEAN) + thd(i,k,j)*rho_d(i,k,j) )/(local_count+1.)/rho_mean
            hist_mean(i,k,j,P_THX_MEAN) = (local_count*hist_mean(i,k,j,P_THX_MEAN) + thetax*rhox )/(local_count+1.)/rhox_mean
            hist_mean(i,k,j,P_THY_MEAN) = (local_count*hist_mean(i,k,j,P_THY_MEAN) + thetay*rhoy )/(local_count+1.)/rhoy_mean
            hist_mean_8w(i,k,j,P_THZ_MEAN) = (local_count*hist_mean_8w(i,k,j,P_THZ_MEAN) + thetaz*rhoz )/(local_count+1.)/rhoz_mean
            hist_mean(i,k,j,P_THTH_MEAN) = (local_count*hist_mean(i,k,j,P_THTH_MEAN) + thd(i,k,j)**2 )/(local_count+1.)

            hist_mean(i,k,j,P_Q_MEAN) = (local_count*hist_mean(i,k,j,P_Q_MEAN) + Q(i,k,j)*rho_d(i,k,j) )/(local_count+1.)/rho_mean
            hist_mean(i,k,j,P_QX_MEAN) = (local_count*hist_mean(i,k,j,P_QX_MEAN) + qx*rhox )/(local_count+1.)/rhox_mean
            hist_mean(i,k,j,P_QY_MEAN) = (local_count*hist_mean(i,k,j,P_QY_MEAN) + qy*rhoy )/(local_count+1.)/rhoy_mean
            hist_mean_8w(i,k,j,P_QZ_MEAN) = (local_count*hist_mean_8w(i,k,j,P_QZ_MEAN) + qz*rhoz )/(local_count+1.)/rhoz_mean
            hist_mean(i,k,j,P_QQ_MEAN) = (local_count*hist_mean(i,k,j,P_QQ_MEAN) + Q(i,k,j)*2 )/(local_count+1.)

            !mean total fluxes
            !TODO: momentum fluxes from advection module

            hist_mean(i,k,j,P_RUTH_TOT_MEAN)  = (local_count*hist_mean(i,k,j,P_RUTH_TOT_MEAN) + ruth_tot(i,k,j) )/(local_count+1.)
            hist_mean(i,k,j,P_RVTH_TOT_MEAN)  = (local_count*hist_mean(i,k,j,P_RVTH_TOT_MEAN) + rvth_tot(i,k,j) )/(local_count+1.)
            hist_mean_8w(i,k,j,P_WWTH_TOT_MEAN)  = (local_count*hist_mean_8w(i,k,j,P_WWTH_TOT_MEAN) + wwth_tot(i,k,j) )/(local_count+1.)

            hist_mean(i,k,j,P_RUQ_TOT_MEAN)  = (local_count*hist_mean(i,k,j,P_RUQ_TOT_MEAN) + ruq_tot(i,k,j) )/(local_count+1.)
            hist_mean(i,k,j,P_RVQ_TOT_MEAN)  = (local_count*hist_mean(i,k,j,P_RVQ_TOT_MEAN) + rvq_tot(i,k,j) )/(local_count+1.)
            hist_mean_8w(i,k,j,P_WWQ_TOT_MEAN)  = (local_count*hist_mean_8w(i,k,j,P_WWQ_TOT_MEAN) + wwq_tot(i,k,j) )/(local_count+1.)

            !mean correction terms
            hist_mean_8w(i,k,j,P_CORR_UTH)  = (local_count*hist_mean_8w(i,k,j,P_CORR_UTH) + corr_uth )/(local_count+1.)
            hist_mean_8w(i,k,j,P_CORR_VTH)  = (local_count*hist_mean_8w(i,k,j,P_CORR_VTH) + corr_vth )/(local_count+1.)
            hist_mean_8w(i,k,j,P_CORR_DTHDT)  = (local_count*hist_mean_8w(i,k,j,P_CORR_DTHDT) + corr_dthdt )/(local_count+1.)

            hist_mean_8w(i,k,j,P_CORR_UQ)  = (local_count*hist_mean_8w(i,k,j,P_CORR_UQ) + corr_uq )/(local_count+1.)
            hist_mean_8w(i,k,j,P_CORR_VQ)  = (local_count*hist_mean_8w(i,k,j,P_CORR_VQ) + corr_vq )/(local_count+1.)
            hist_mean_8w(i,k,j,P_CORR_DQDT)  = (local_count*hist_mean_8w(i,k,j,P_CORR_DQDT) + corr_dqdt )/(local_count+1.)

          ENDIF

          hist_mean(i,k,j,P_T_TEND_RADLW_MEAN)  = (local_count*hist_mean(i,k,j,P_T_TEND_RADLW_MEAN) + t_tend_radlw(i,k,j) )/(local_count+1.)
          hist_mean(i,k,j,P_T_TEND_RADSW_MEAN)  = (local_count*hist_mean(i,k,j,P_T_TEND_RADSW_MEAN) + t_tend_radsw(i,k,j) )/(local_count+1.)
          hist_mean(i,k,j,P_T_TEND_MP_MEAN)  = (local_count*hist_mean(i,k,j,P_T_TEND_MP_MEAN) + t_tend_mp(i,k,j) )/(local_count+1.)
          hist_mean(i,k,j,P_Q_TEND_MP_MEAN)  = (local_count*hist_mean(i,k,j,P_Q_TEND_MP_MEAN) + qv_tend_mp(i,k,j) )/(local_count+1.)
          !TODO: temperature tendencies from cumulus scheme; small time step tendencies
          IF (config_flags%output_sgs_fluxes==1) THEN
            hist_mean(i,k,j,P_sgs_UU_mean)  = (local_count*hist_mean(i,k,j,P_sgs_UU_mean)  + sgs_fluxes(i,k,j,P_sgs_UU) )  / (local_count+1.)
            hist_mean(i,k,j,P_sgs_VV_mean)  = (local_count*hist_mean(i,k,j,P_sgs_VV_mean)  + sgs_fluxes(i,k,j,P_sgs_VV)  )  / (local_count+1.)
            hist_mean(i,k,j,P_sgs_WW_mean)  = (local_count*hist_mean(i,k,j,P_sgs_WW_mean)  + sgs_fluxes(i,k,j,P_sgs_WW)  )  / (local_count+1.)
            hist_mean(i,k,j,P_sgs_UV_mean)  = (local_count*hist_mean(i,k,j,P_sgs_UV_mean)  + sgs_fluxes(i,k,j,P_sgs_UV) )  / (local_count+1.)
            hist_mean_8w(i,k,j,P_sgs_UW_mean)  = (local_count*hist_mean_8w(i,k,j,P_sgs_UW_mean)  + sgs_fluxes_8w(i,k,j,P_sgs_UW) )  / (local_count+1.)
            hist_mean_8w(i,k,j,P_sgs_VW_mean)  = (local_count*hist_mean_8w(i,k,j,P_sgs_VW_mean)  + sgs_fluxes_8w(i,k,j,P_sgs_VW) )  / (local_count+1.)
            hist_mean_8w(i,k,j,P_sgs_WU_mean)  = (local_count*hist_mean_8w(i,k,j,P_sgs_WU_mean)  + sgs_fluxes_8w(i,k,j,P_sgs_WU) )  / (local_count+1.)
            hist_mean_8w(i,k,j,P_sgs_WV_mean)  = (local_count*hist_mean_8w(i,k,j,P_sgs_WV_mean)  + sgs_fluxes_8w(i,k,j,P_sgs_WV) )  / (local_count+1.)

            hist_mean(i,k,j,P_sgs_UTH_mean) = (local_count*hist_mean(i,k,j,P_sgs_UTH_mean) + sgs_fluxes(i,k,j,P_sgs_UTH) ) / (local_count+1.)
            hist_mean(i,k,j,P_sgs_VTH_mean) = (local_count*hist_mean(i,k,j,P_sgs_VTH_mean) + sgs_fluxes(i,k,j,P_sgs_VTH) ) / (local_count+1.)
            hist_mean_8w(i,k,j,P_sgs_WTH_mean) = (local_count*hist_mean_8w(i,k,j,P_sgs_WTH_mean) + sgs_fluxes_8w(i,k,j,P_sgs_WTH) ) / (local_count+1.)

            hist_mean(i,k,j,P_sgs_UQ_mean) = (local_count*hist_mean(i,k,j,P_sgs_UQ_mean) + sgs_fluxes(i,k,j,P_sgs_UQ) ) / (local_count+1.)
            hist_mean(i,k,j,P_sgs_VQ_mean) = (local_count*hist_mean(i,k,j,P_sgs_VQ_mean) + sgs_fluxes(i,k,j,P_sgs_VQ) ) / (local_count+1.)
            hist_mean_8w(i,k,j,P_sgs_WQ_mean) = (local_count*hist_mean_8w(i,k,j,P_sgs_WQ_mean) + sgs_fluxes_8w(i,k,j,P_sgs_WQ) ) / (local_count+1.)
          ENDIF

        end DO
      end DO
    end DO
    endif

    if (config_flags%do_avgflx_em .eq. 1) then
    DO j=jts,jte
       DO k=kts,kte
          DO i=its,ite
             avgflx_rum(i,k,j) = (local_count*avgflx_rum(i,k,j) + ru_m(i,k,j))/(local_count+1.)
             avgflx_rvm(i,k,j) = (local_count*avgflx_rvm(i,k,j) + rv_m(i,k,j))/(local_count+1.)
             avgflx_wwm(i,k,j) = (local_count*avgflx_wwm(i,k,j) + ww_m(i,k,j))/(local_count+1.)
          end DO
       end DO
    end DO
    endif

    if (do_cu .and. &
         & present(avgflx_cfu1) .and. present(avgflx_cfd1) .and. present(avgflx_dfu1) &
         & .and. present(avgflx_efu1) .and. present(avgflx_dfd1) .and. present(avgflx_efd1) &
         & .and. present(cfu1) .and. present(cfd1) .and. present(dfu1) &
         & .and. present(efu1) .and. present(dfd1) .and. present(efd1) ) then
       DO j=jts,jte
          DO k=kts,kte
             DO i=its,ite
                avgflx_cfu1(i,k,j) = (local_count*avgflx_cfu1(i,k,j) + &
                     & cfu1(i,k,j)) / (local_count+1.)
                avgflx_cfd1(i,k,j) = (local_count*avgflx_cfd1(i,k,j) + &
                     & cfd1(i,k,j)) / (local_count+1.)
                avgflx_dfu1(i,k,j) = (local_count*avgflx_dfu1(i,k,j) + &
                     & dfu1(i,k,j)) / (local_count+1.)
                avgflx_efu1(i,k,j) = (local_count*avgflx_efu1(i,k,j) + &
                     & efu1(i,k,j)) / (local_count+1.)
                avgflx_dfd1(i,k,j) = (local_count*avgflx_dfd1(i,k,j) + &
                     & dfd1(i,k,j)) / (local_count+1.)
                avgflx_efd1(i,k,j) = (local_count*avgflx_efd1(i,k,j) + &
                     & efd1(i,k,j)) / (local_count+1.)
             end DO
          end DO
       end DO
    end if

    return
  end subroutine upd_avgflx
end MODULE module_avgflx_em
