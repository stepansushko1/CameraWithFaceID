package com.example.faceapp;

import androidx.annotation.NonNull;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.util.ArrayList;
import java.util.List;

public class DatabaseHelper {
    private FirebaseDatabase db;
    private DatabaseReference db_ref;
    private List<String> names = new ArrayList<>();

    public DatabaseHelper(){
        db = FirebaseDatabase.getInstance();
        db_ref = db.getReference("User");

    }
    public void read_users(){
        db_ref.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                names.clear();
                if (snapshot.exists()){

                }

            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {

            }
        });
    }

}
