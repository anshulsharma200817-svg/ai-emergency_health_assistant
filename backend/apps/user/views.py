# pyrefly: ignore [missing-import]
import logging
# pyrefly: ignore [missing-import]
from rest_framework.views import APIView
# pyrefly: ignore [missing-import]
from rest_framework.response import Response
# pyrefly: ignore [missing-import]
from rest_framework import status

from .serializers import UserRegisterSerializer

logger = logging.getLogger(__name__)


class RegisterView(APIView):

    def post(self, request):
        serializer = UserRegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()

            # Output registration notice directly to server terminal
            terminal_msg = (
                "\n" + "=" * 60 + "\n"
                f"🎉 [REGISTER SUCCESS] New User Registered!\n"
                f"   ID:          {user.id}\n"
                f"   Email:       {user.email}\n"
                f"   First Name:  {user.first_name or '(None)'}\n"
                f"   Last Name:   {user.last_name or '(None)'}\n"
                f"   Date Joined: {user.date_joined}\n"
                + "=" * 60 + "\n"
            )
            print(terminal_msg, flush=True)
            logger.info(f"User registered successfully: {user.email}")

            return Response(serializer.data, status=status.HTTP_201_CREATED)

        # Output failure notice to server terminal
        error_msg = f"❌ [REGISTER FAILED] Validation errors: {serializer.errors}"
        print(f"\n{error_msg}\n", flush=True)
        logger.warning(error_msg)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)