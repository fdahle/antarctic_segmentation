--
-- PostgreSQL database dump
--

-- Dumped from database version 12.11 (Ubuntu 12.11-1.pgdg20.04+1)
-- Dumped by pg_dump version 12.11 (Ubuntu 12.11-1.pgdg20.04+1)

-- Started on 2022-06-13 09:10:53 CEST

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 3882 (class 1262 OID 16384)
-- Name: antarctica; Type: DATABASE; Schema: -; Owner: -
--

CREATE DATABASE antarctica WITH TEMPLATE = template0 ENCODING = 'UTF8' LC_COLLATE = 'en_US.UTF-8' LC_CTYPE = 'en_US.UTF-8';


\connect antarctica

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 2 (class 3079 OID 16402)
-- Name: postgis; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public;


SET default_table_access_method = heap;

--
-- TOC entry 209 (class 1259 OID 25619)
-- Name: cameras; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cameras (
    camera_id bigint NOT NULL,
    camera_name text,
    camera_type text,
    lens_type text,
    focal_length_nominal numeric(8,4),
    focal_length_calibrated numeric(8,4),
    corner_fid_x numeric(6,4),
    corner_fid_y numeric(6,4),
    midside_fid_x numeric(6,4),
    midside_fid_y numeric(6,4),
    ppc_calibrated_x numeric(8,4),
    ppc_calibrated_y numeric(8,4),
    fid_1_x numeric(8,4),
    fid_1_y numeric(8,4),
    fid_2_x numeric(8,4),
    fid_2_y numeric(8,4),
    fid_3_x numeric(8,4),
    fid_3_y numeric(8,4),
    fid_4_x numeric(8,4),
    fid_4_y numeric(8,4),
    fid_5_x numeric(8,4),
    fid_5_y numeric(8,4),
    fid_6_x numeric(8,4),
    fid_6_y numeric(8,4),
    fid_7_x numeric(8,4),
    fid_7_y numeric(8,4),
    fid_8_x numeric(8,4),
    fid_8_y numeric(8,4),
    distance_fid_1_2 numeric(6,3),
    distance_fid_3_4 numeric(6,3),
    distance_fid_5_6 numeric(6,3),
    distance_fid_7_8 numeric(6,3),
    distance_fid_1_3 numeric(6,3),
    distance_fid_2_3 numeric(6,3),
    distance_fid_1_4 numeric(6,3),
    distance_fid_2_4 numeric(6,3),
    angle_fid_12_34 text,
    angle_fid_56_78 text
);

--
-- TOC entry 208 (class 1259 OID 17419)
-- Name: images_segmentation; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.images_segmentation (
    image_id text NOT NULL,
    labeled_by text,
    perc_ice numeric(6,3),
    perc_snow numeric(6,3),
    perc_rocks numeric(6,3),
    perc_water numeric(6,3),
    perc_clouds numeric(6,3),
    perc_sky numeric(6,3),
    perc_other numeric(6,3),
    model_name text,
    improvement_applied boolean,
    last_change timestamp without time zone
);

--
-- TOC entry 210 (class 1259 OID 25627)
-- Name: images; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.images (
    image_id text NOT NULL,
    file_path text,
    mask_path text,
    tma_number numeric(4,0),
    roll text,
    frame numeric(4,0),
    view_direction text,
    altitude numeric(10,3),
    azimuth numeric(8,4),
    flying_direction text,
    date_of_recording text,
    id_cam numeric(3,0),
    location text,
    x_coords numeric(12,2),
    y_coords numeric(12,2),
    point_x numeric(5,2),
    point_y numeric(5,2),
    coordinates public.geometry,
    footprint public.geometry,
    comment text,
    last_change timestamp without time zone
);

--
-- TOC entry 211 (class 1259 OID 25648)
-- Name: images_properties; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.images_properties (
    image_id text NOT NULL,
    logo_removed boolean DEFAULT false,
    right_orientation text,
    image_width numeric(6,0),
    image_height numeric(6,0),
    mask_size numeric(3,0),
    subset_width numeric(5,0),
    subset_height numeric(5,0),
    subset_n_x numeric(6,0),
    subset_n_y numeric(6,0),
    subset_e_x numeric(6,0),
    subset_e_y numeric(6,0),
    subset_s_x numeric(6,0),
    subset_s_y numeric(6,0),
    subset_w_x numeric(6,0),
    subset_w_y numeric(6,0),
    subset_extraction_date date,
    fid_type numeric(2,0),
    fid_mark_1_x numeric(6,0),
    fid_mark_1_y numeric(6,0),
    fid_mark_1_estimated boolean,
    fid_mark_2_x numeric(6,0),
    fid_mark_2_y numeric(6,0),
    fid_mark_2_estimated boolean,
    fid_mark_3_x numeric(6,0),
    fid_mark_3_y numeric(6,0),
    fid_mark_3_estimated boolean,
    fid_mark_4_x numeric(6,0),
    fid_mark_4_y numeric(6,0),
    fid_mark_4_estimated boolean,
    fid_mark_5_x numeric(6,0),
    fid_mark_5_y numeric(6,0),
    fid_mark_5_estimated boolean,
    fid_mark_6_x numeric(6,0),
    fid_mark_6_y numeric(6,0),
    fid_mark_6_estimated boolean,
    fid_mark_7_x numeric(6,0),
    fid_mark_7_y numeric(6,0),
    fid_mark_7_estimated boolean,
    fid_mark_8_x numeric(6,0),
    fid_mark_8_y numeric(6,0),
    fid_mark_8_estimated boolean,
    fid_extraction_date date,
    ppa_x numeric(6,0),
    ppa_y numeric(6,0),
    comment text,
    last_change timestamp without time zone
);

--
-- TOC entry 216 (class 1259 OID 33933)
-- Name: images_tie_points; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.images_tie_points (
    id integer NOT NULL,
    image_1_id text,
    image_2_id text,
    method text,
    extraction_date date,
    number_tie_points numeric,
    average_quality numeric,
    tie_points text
);


--
-- TOC entry 215 (class 1259 OID 33931)
-- Name: images_tie_points_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.images_tie_points_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- TOC entry 3883 (class 0 OID 0)
-- Dependencies: 215
-- Name: images_tie_points_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.images_tie_points_id_seq OWNED BY public.images_tie_points.id;

--
-- TOC entry 3729 (class 2604 OID 33936)
-- Name: images_tie_points id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.images_tie_points ALTER COLUMN id SET DEFAULT nextval('public.images_tie_points_id_seq'::regclass);


--
-- TOC entry 3735 (class 2606 OID 25626)
-- Name: cameras cameras_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cameras
    ADD CONSTRAINT cameras_pkey PRIMARY KEY (camera_id);

--
-- TOC entry 3737 (class 2606 OID 25634)
-- Name: images images_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.images
    ADD CONSTRAINT images_pkey PRIMARY KEY (image_id);


--
-- TOC entry 3739 (class 2606 OID 25655)
-- Name: images_properties images_properties_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.images_properties
    ADD CONSTRAINT images_properties_pkey PRIMARY KEY (image_id);


--
-- TOC entry 3733 (class 2606 OID 17426)
-- Name: images_segmentation images_segmentation_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.images_segmentation
    ADD CONSTRAINT images_segmentation_pkey PRIMARY KEY (image_id);


--
-- TOC entry 3743 (class 2606 OID 33941)
-- Name: images_tie_points images_tie_points_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.images_tie_points
    ADD CONSTRAINT images_tie_points_pkey PRIMARY KEY (id);


-- Completed on 2022-06-13 09:10:53 CEST

--
-- PostgreSQL database dump complete
--

