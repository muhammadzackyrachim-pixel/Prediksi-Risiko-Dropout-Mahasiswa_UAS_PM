\# Data Dictionary - Student Dropout Prediction



\## 1. Deskripsi Dataset

Dataset ini berisi data mahasiswa yang digunakan untuk memprediksi risiko dropout. 

Setiap baris merepresentasikan satu mahasiswa dengan atribut demografis, akademik, dan kondisi ekonomi.



\- Jumlah data: 4424 baris

\- Jumlah kolom asli: 37

\- Target asli: `Target`

\- Target penelitian: `dropout\_risk`



\## 2. Definisi Target

Kolom target penelitian dibuat dari kolom `Target`:

\- `1` = Dropout

\- `0` = Non-dropout (Graduate / Enrolled)



\## 3. Data Dictionary

| No | Nama Kolom | Tipe Data | Deskripsi | Contoh Nilai | Peran |

|----|------------|-----------|-----------|--------------|-------|

| 1 | Marital status | int | Status pernikahan mahasiswa | 1, 2, 3 | Fitur |

| 2 | Application mode | int | Jalur atau mode pendaftaran mahasiswa | 1, 17, 39 | Fitur |

| 3 | Application order | int | Urutan pilihan program studi saat mendaftar | 1, 2, 5 | Fitur |

| 4 | Course | int | Kode program studi yang diambil mahasiswa | 171, 9254 | Fitur |

| 5 | Daytime/evening attendance | int | Waktu perkuliahan mahasiswa (siang/malam) | 0, 1 | Fitur |

| 6 | Previous qualification | int | Kualifikasi pendidikan sebelumnya | 1, 2, 3 | Fitur |

| 7 | Previous qualification (grade) | float | Nilai dari pendidikan sebelumnya | 122.0, 160.0 | Fitur |

| 8 | Nacionality | int | Kode kewarganegaraan mahasiswa | 1, 2 | Fitur |

| 9 | Mother's qualification | int | Kualifikasi pendidikan ibu | 1, 19, 37 | Fitur |

| 10 | Father's qualification | int | Kualifikasi pendidikan ayah | 3, 12, 38 | Fitur |

| 11 | Mother's occupation | int | Kode pekerjaan ibu | 3, 5, 9 | Fitur |

| 12 | Father's occupation | int | Kode pekerjaan ayah | 3, 9 | Fitur |

| 13 | Admission grade | float | Nilai saat masuk perguruan tinggi | 119.6, 142.5 | Fitur |

| 14 | Displaced | int | Status mahasiswa pindahan/terdampak kondisi tertentu | 0, 1 | Fitur |

| 15 | Educational special needs | int | Indikasi kebutuhan pendidikan khusus | 0, 1 | Fitur |

| 16 | Debtor | int | Status mahasiswa memiliki tunggakan/utang | 0, 1 | Fitur |

| 17 | Tuition fees up to date | int | Status pembayaran uang kuliah tepat waktu | 0, 1 | Fitur |

| 18 | Gender | int | Jenis kelamin mahasiswa | 0, 1 | Fitur |

| 19 | Scholarship holder | int | Status penerima beasiswa | 0, 1 | Fitur |

| 20 | Age at enrollment | int | Umur mahasiswa saat masuk | 17, 18, 25, dst | Fitur |

| 21 | International | int | Status mahasiswa internasional | 0, 1 | Fitur |

| 22 | Curricular units 1st sem (credited) | int | Jumlah SKS/mata kuliah semester 1 yang diakui | 0, 1, 2 | Fitur |

| 23 | Curricular units 1st sem (enrolled) | int | Jumlah mata kuliah semester 1 yang diambil | 5, 6, 7 | Fitur |

| 24 | Curricular units 1st sem (evaluations) | int | Jumlah evaluasi/ujian semester 1 | 0, 6, 8 | Fitur |

| 25 | Curricular units 1st sem (approved) | int | Jumlah mata kuliah semester 1 yang lulus | 0, 5, 6 | Fitur |

| 26 | Curricular units 1st sem (grade) | float | Rata-rata nilai semester 1 | 0.0, 12.5, 14.2 | Fitur |

| 27 | Curricular units 1st sem (without evaluations) | int | Jumlah mata kuliah semester 1 tanpa evaluasi | 0, 1 | Fitur |

| 28 | Curricular units 2nd sem (credited) | int | Jumlah SKS/mata kuliah semester 2 yang diakui | 0, 1, 2 | Fitur |

| 29 | Curricular units 2nd sem (enrolled) | int | Jumlah mata kuliah semester 2 yang diambil | 5, 6, 7 | Fitur |

| 30 | Curricular units 2nd sem (evaluations) | int | Jumlah evaluasi/ujian semester 2 | 0, 6, 8 | Fitur |

| 31 | Curricular units 2nd sem (approved) | int | Jumlah mata kuliah semester 2 yang lulus | 0, 5, 6 | Fitur |

| 32 | Curricular units 2nd sem (grade) | float | Rata-rata nilai semester 2 | 0.0, 12.8, 14.0 | Fitur |

| 33 | Curricular units 2nd sem (without evaluations) | int | Jumlah mata kuliah semester 2 tanpa evaluasi | 0, 1 | Fitur |

| 34 | Unemployment rate | float | Tingkat pengangguran saat periode data dikumpulkan | 7.6, 11.1 | Fitur |

| 35 | Inflation rate | float | Tingkat inflasi saat periode data dikumpulkan | 0.6, 1.4 | Fitur |

| 36 | GDP | float | Produk Domestik Bruto saat periode data dikumpulkan | 0.32, 1.74 | Fitur |

| 37 | Target | object | Label asli status mahasiswa | Dropout / Graduate / Enrolled | Target Asli |

| 38 | dropout\_risk | int | Target biner hasil transformasi dari `Target` | 0 / 1 | Target Penelitian |



\## 4. Catatan Preprocessing

\- Kolom `Target` diubah menjadi `dropout\_risk`.

\- `dropout\_risk = 1` jika `Target = Dropout`

\- `dropout\_risk = 0` jika `Target = Graduate` atau `Enrolled`

\- Seluruh fitur numerik digunakan sebagai input model machine learning.

\- Dataset tidak memiliki missing value dan duplikat berdasarkan hasil eksplorasi awal.



\## 5. Catatan Penting

Beberapa kolom seperti `Course`, `Application mode`, `Mother's qualification`, `Father's qualification`, `Mother's occupation`, dan `Father's occupation` merupakan \*\*kode kategori\*\*, sehingga pada tahap modeling dapat diperlakukan sebagai fitur kategorikal terenkode.

